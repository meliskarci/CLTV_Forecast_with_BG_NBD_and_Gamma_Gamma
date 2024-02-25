############   CLTV Forecast with BG/NBD and Gamma-Gamma   ############

# Y shoes company wants to determine a roadmap for sales and marketing activities. In order for the company to make medium-long term
# plans, it is necessary to estimate the potential value that existing customers will provide to the company in the
# future. The data set consists of information obtained from the past shopping behavior of customers who made their
# last purchases from Y shoes company via OmniChannel (both online and offline shopping) in 2020 - 2021.

# - master_id:   Unique customer number
# - order_channel: Which channel of the shopping platform is used (Android, iOS, Desktop, Mobile)
# - last_order_channel: Channel where last purchase was made
# - first_order_date: Date of the customer's first purchase
# - last_order_date: Customer's last purchase date
# - last_order_date_online: The customer's last shopping date on the online platform
# - last_order_date_offline: The last shopping date of the customer on the offline platform
# - order_num_total_ever_online: Total number of purchases made by the customer on the online platform
# - order_num_total_ever_offline: Total number of purchases made by the customer offline
# - customer_value_total_ever_offline: Total fee paid by the customer for offline purchases
# - customer_value_total_ever_online: Total fee paid by the customer for online shopping
# - interested_in_categories_12: List of categories the customer has shopped in the last 12 months

#!pip install lifetimes
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None


# --------------- Data Preparing ---------------

df_ = pd.read_csv("Projects/dataset.csv")
df = df_.copy()
df.head()
df.describe().T
df.isnull().sum()
df.info()

# Let's define the "outlier_thresholds" and "replace_with_thresholds" functions required to suppress outliers.

# It determines the threshold value for the variable entered into it.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Suppresses outliers.
# Note: When calculating cltv, frequency values must be integer. Therefore, we will round the lower and upper limits with round().
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

# Let's suppress the variables "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline", "customer_value_total_ever_online" if there are outliers.
columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
           "customer_value_total_ever_offline","customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df, col)

# Omnichannel means that customers shop both online and offline platforms.
# We create new variables for each customer's total number of purchases and spending.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# When we examined the variable types, we found that the type of variables expressing date is object. We convert their
# type to date.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()


# --------------- Creating the CLTV Data Structure ---------------

# We take 2 days after the date of the last purchase in the data set as the analysis date.
df["last_order_date"].max()

analysis_date = dt.datetime(2021,6,1)

# We will create a new dataframe named cltv_df containing customer_id, recency, T_weekly, frequency and monetary values.
#
# - recency: Time since last purchase per user (weekly)
# - T_weekly: Customer's age. How long before the analysis date was the first purchase (Weekly)
# - frequency: Total number of recurring purchases (frequency must be >1 to become our customer)
# - monetary: Average earnings per purchase

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency"] = (df["last_order_date"] - df["first_order_date"]) / pd.Timedelta(days=7)
cltv_df["T_weekly"] = (analysis_date - df["first_order_date"]) / pd.Timedelta(days=7)
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()


# --------------- BG/NBD, Establishment of Gamma-Gamma Models, Calculation of 6-month CLTV ---------------

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T_weekly'])

# Let's estimate the expected purchases from customers within 3 months and add it to the cltv dataframe
# as exp_sales_3_month.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T_weekly'])

# If we want to see the total number of purchases expected from customers within 3 months
bgf.predict(4*3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T_weekly']).sum()

# Let's estimate the expected purchases from customers within 6 months and add it to the cltv dataframe
# as exp_sales_6_month.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T_weekly'])

# Let's examine the 10 people who will make the most purchases in 3 and 6 months.

# Top 10 people expected to buy the most in 3 months and expected unit values
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

# Top 10 people expected to buy the most in 6 months and expected unit values
cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

# In the forecast for the number of purchases for 3 and 6 months for 2020-2021 Omnichannel customers, the top 10
# observations with the highest expected return are the same.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df.head()
# With “exp_average_value” we found the expected profitability of each customer.

# We will connect the BG-NBD model to the Gamma-Gamma model to find the 6-month CLTV value.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv

# Let's observe the 20 people with the highest CLTV values. In other words, the ones that give us the highest return.
cltv_df.sort_values("cltv",ascending=False)[:10]


# --------------- Creating Segments Based on CLTV ---------------

# We divide all our customers into 4 groups (segments) according to 6-month CLTV and add the group names to the data set.
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.head(20)

# Does it make sense to divide customers into 4 groups based on CLTV scores? Should there be less or more?
#
# When we examine the 0th and 14th observations, they are both in the 'A' segment, but there seems to be a high
# difference between the two. Maybe segment 'A' can be divided into segments according to the rate situation. The 17th
# observation is in the 'A' segment and the 18th observation is in the B segment, but at values close to each other.
# Investigations can be made regarding such observations.

####  You can read the suggestions for the conclusion we made as a result of our study in the "read.me" section. ####
