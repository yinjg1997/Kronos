import json

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(kline_df['timestamps'], close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(kline_df['timestamps'], close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(kline_df['timestamps'], volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(kline_df['timestamps'], volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.xticks(rotation=45, ha='right')  # 旋转45度，右对齐
    plt.tight_layout()
    plt.show()


def plot_prediction2(kline_df, pred_df):
    half = (len(kline_df) + 1) // 2
    kline_df = kline_df.iloc[half:]

    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(kline_df['timestamps'], close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(kline_df['timestamps'], close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(kline_df['timestamps'], volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(kline_df['timestamps'], volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    # 7. 核心：设置x轴每天一个间隔
    # 7.1 主刻度：每天显示一个（间隔1天）
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # interval=1 表示每天一个刻度
    # 7.2 日期格式：显示“年-月-日”（适合日线级数据）
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xticks(rotation=45, ha='right')  # 旋转45度，右对齐
    plt.tight_layout()
    plt.show()


def get_input_data(lookback: int, pred_len: int,is_compare: bool) -> pd.DataFrame:
    """
    将接口数据处理为 Kronos 可用的数据
    """
    with open('./data/futures_data.json', 'r') as f:
        futures_data = json.load(f)

    datas = futures_data['data']
    if is_compare:
        datas = datas[-lookback-pred_len:]
    else:
        datas = datas[-lookback]

    print(len(datas))
    # 将JSON数据转换为DataFrame
    df = pd.DataFrame(datas)

    # 补充 timestamps 字段
    df["timestamps"] = pd.to_datetime(df["date"])
    # 补充缺失字段
    df["amount"] = 0

    if not is_compare:
        # 根据现有日期生成之后的120个日期, 为预测留 120 个位置 pred_len
        last_date = df["timestamps"].iloc[-1]  # 获取最后一个日期
        new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)
        new_rows = pd.DataFrame({
            "timestamps": new_dates
        })
        # 3. 合并到原DataFrame
        df = pd.concat([df, new_rows], ignore_index=True)

    return df


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. Prepare Data
lookback = 400
pred_len = 120

df = get_input_data(lookback, pred_len, is_compare=True)
df['timestamps'] = pd.to_datetime(df['timestamps'])

x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback - 1, 'timestamps']
print(f"x_timestamp.tail: ")
print(x_timestamp.tail())
y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']
print(f"y_timestamp.head: ")
print(y_timestamp.head())

# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback + pred_len - 1]

# visualize
plot_prediction2(kline_df, pred_df)
