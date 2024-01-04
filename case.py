###########################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
###########################################################

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın Fonksiyonlaştırılması


###########################################################
# 1. Verinin Hazırlanması (Data Preperation)
###########################################################

# İngiltere merkezli perakende şirketi satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin
# 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış işlemlerini içeriyor.
# Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.

# Değişkenler

# Invoice: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi.
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# Price: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


###########################################################
# Gerekli Kütüphane ve Fonksiyonlar
###########################################################

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%3f." % x)
%matplotlib inline

###########################################################
# Verinin Okunması
###########################################################

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

###########################################################
# Veri İlk İzlenim
###########################################################

df.shape
df.info()
df.describe().T
df.isnull().sum()

###########################################################
# Veri Hazırlama
###########################################################

df.dropna(inplace=True)
df.isnull().sum()
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df["Invoice"] = df["Invoice"].astype(str)
df = df[~df["Invoice"].str.contains("C", na=False)]

def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q3 - 1.5 * iqr

    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


columns = ["Quantity", "Price"]

for col in columns:
    replace_with_thresholds(df, col)

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]


###########################################################
# Lifetime Veri Yapısının Hazırlanması
###########################################################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
                                         "Invoice": lambda Invoice: Invoice.nunique(),
                                         "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df.describe().T

cltv_df = cltv_df[cltv_df["frequency"] > 1]

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7


###########################################################
# BG-NBD Modelinin Kurulması
###########################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

cltv_df["expected_purc_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(24,
                                                                                           cltv_df["frequency"],
                                                                                           cltv_df["recency"],
                                                                                           cltv_df["T"])

###########################################################
# Tahmin Sonuçlarının Değerlendirilmesi
###########################################################

plot_period_transactions(bgf)
plt.show()

###########################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
###########################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df['expected_average_profit'] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df["cltv_6_month"] = ggf.customer_lifetime_value(bgf,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"],
                                                   cltv_df["monetary"],
                                                   time=6,
                                                   discount_rate=0.01,
                                                   freq="W")

cltv_df.head()
cltv_df.sort_values(by="cltv_6_month", ascending=False).head(10)

cltv_df["cltv_1_month"] = ggf.customer_lifetime_value(bgf,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"],
                                                   cltv_df["monetary"],
                                                   time=1,
                                                   discount_rate=0.01,
                                                   freq="W")

cltv_df["cltv_12_month"] = ggf.customer_lifetime_value(bgf,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"],
                                                   cltv_df["monetary"],
                                                   time=12,
                                                   discount_rate=0.01,
                                                   freq="W")

cltv_df.sort_values(by="cltv_1_month", ascending=False).head(10)
cltv_df.sort_values(by="cltv_12_month", ascending=False).head(10)


###########################################################
# CLTV'ye Göre Segmentlerin Oluşturulması
###########################################################

cltv_df["segment"] = pd.qcut(cltv_df["cltv_6_month"], 4, labels=["D", "C", "B", "A"])
cltv_df.head(10)

cltv_df.groupby("segment").agg(["mean", "count", "sum"])


###########################################################
# Çalışmanın Fonksiyonlaştırılması
###########################################################

def create_cltv_p(dataframe, month=6):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Price"] * dataframe["Quantity"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                                    lambda date: (today_date - date.min()).days],
                                                    'Invoice': lambda num: num.nunique(),
                                                    'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[cltv_df["frequency"] > 1]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])


    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df["frequency"],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_6_month"] = bgf.predict(24,
                                                   cltv_df["frequency"],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_1_year"] = bgf.predict(48,
                                                  cltv_df["frequency"],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df['expected_average_profit'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBF ve GG modeli ile CLTV'nin hesaplanması
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi
                                       discount_rate=0.01)
    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, how="left", on='Customer ID')

    # 5. CLTV'ye Göre Segmentlerin Oluşturulması
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return  cltv_final

create_cltv_p(df)

























