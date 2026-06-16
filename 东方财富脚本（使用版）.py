import os
import pandas as pd
import akshare as ak
from openai import OpenAI
import requests
from datetime import datetime

# ====================== 配置区域 ======================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SERVERCHAN_SENDKEY = os.getenv("SERVERCHAN_SENDKEY")
MODEL_NAME = "deepseek-v4-pro"

# 关注标的股票代码（6位数字，可自行增减）
WATCHLIST_STOCKS = ["000796", "601888", "002797"]
# 各榜单展示数量
TOP_N = 5
# =====================================================

# 启动前置校验：避免密钥为空时报错不明确
if not DEEPSEEK_API_KEY:
    print("❌ 错误：未读取到 DEEPSEEK_API_KEY 环境变量，请检查GitHub Secrets配置")
    exit(1)
if not SERVERCHAN_SENDKEY:
    print("❌ 错误：未读取到 SERVERCHAN_SENDKEY 环境变量，请检查GitHub Secrets配置")
    exit(1)

def get_index_data():
    """获取四大核心大盘指数实时表现"""
    try:
        index_df = ak.stock_zh_index_spot_em()
        if index_df.empty:
            return "暂无指数行情数据"
        # 筛选核心指数：上证指数、深证成指、创业板指、科创50
        core_index_codes = ["000001", "399001", "399006", "000688"]
        core_df = index_df[index_df["代码"].isin(core_index_codes)]
        
        text = "【核心大盘指数】\n"
        for _, row in core_df.iterrows():
            name = row.get("名称", row.get("指数名称", "未知"))
            point = row.get("最新价", 0)
            change = row.get("涨跌幅", 0)
            amount = row.get("成交额", 0) / 1e8
            text += f"- {name}：{point:.2f} 点，{change:+.2f}%，成交额 {amount:.2f} 亿元\n"
        return text
    except Exception as e:
        return f"指数数据获取失败：{str(e)}"

def get_daily_news():
    """获取东方财富当日财经要闻"""
    try:
        news_df = ak.stock_news_em()
        if news_df.empty:
            return "今日暂无更新的财经新闻"
        recent_news = news_df.head(15)
        news_text = ""
        for _, row in recent_news.iterrows():
            title = row.get("标题", row.get("title", "无标题"))
            pub_time = row.get("发布时间", row.get("time", ""))
            news_text += f"- {title}（{pub_time}）\n"
        return news_text
    except Exception as e:
        return f"新闻获取失败：{str(e)}"

def get_market_data():
    """获取北向资金核心数据"""
    try:
        north_money_df = ak.stock_hsgt_north_net_flow_in_em()
        if north_money_df.empty:
            return "今日休市，暂无北向资金数据"
        latest = north_money_df.iloc[-1]
        data_text = f"""
- 北向资金当日净流入：{latest.get('净流入-北向', '-')} 亿元
- 沪股通净流入：{latest.get('净流入-沪股通', '-')} 亿元
- 深股通净流入：{latest.get('净流入-深股通', '-')} 亿元
        """
        return data_text
    except Exception as e:
        return f"市场数据获取失败：{str(e)}"

def get_industry_data():
    """获取行业板块涨跌幅排名"""
    try:
        industry_df = ak.stock_board_industry_em()
        if industry_df.empty:
            return "暂无行业板块数据"
        industry_sorted = industry_df.sort_values(by="涨跌幅", ascending=False)
        top_up = industry_sorted.head(TOP_N)
        top_down = industry_sorted.tail(TOP_N).iloc[::-1]
        
        text = "【涨幅居前行业】\n"
        for _, row in top_up.iterrows():
            name = row.get("板块名称", "未知")
            change = row.get("涨跌幅", 0)
            leader = row.get("领涨股", "-")
            text += f"- {name}：{change:+.2f}%，领涨：{leader}\n"
        
        text += "\n【跌幅居前行业】\n"
        for _, row in top_down.iterrows():
            name = row.get("板块名称", "未知")
            change = row.get("涨跌幅", 0)
            text += f"- {name}：{change:+.2f}%\n"
        return text
    except Exception as e:
        return f"行业数据获取失败：{str(e)}"

def get_sector_flow():
    """获取行业板块主力资金流向"""
    try:
        flow_df = ak.stock_board_industry_flow_em()
        if flow_df.empty:
            return "暂无板块资金流向数据"
        flow_sorted = flow_df.sort_values(by="主力净流入-净额", ascending=False)
        top_inflow = flow_sorted.head(TOP_N)
        top_outflow = flow_sorted.tail(TOP_N).iloc[::-1]
        
        text = "【主力资金净流入TOP行业】\n"
        for _, row in top_inflow.iterrows():
            name = row.get("名称", "未知")
            net_in = row.get("主力净流入-净额", 0) / 1e8
            text += f"- {name}：主力净流入 {net_in:+.2f} 亿元\n"
        
        text += "\n【主力资金净流出TOP行业】\n"
        for _, row in top_outflow.iterrows():
            name = row.get("名称", "未知")
            net_out = row.get("主力净流入-净额", 0) / 1e8
            text += f"- {name}：主力净流出 {net_out:+.2f} 亿元\n"
        return text
    except Exception as e:
        return f"板块资金流获取失败：{str(e)}"

def get_lhb_data():
    """获取龙虎榜净买卖榜单"""
    try:
        lhb_df = ak.stock_lhb_detail_em()
        if lhb_df.empty:
            return "今日暂无龙虎榜数据"
        lhb_sorted = lhb_df.sort_values(by="净买入额", ascending=False)
        top_buy = lhb_sorted.head(TOP_N)
        top_sell = lhb_sorted.tail(TOP_N).iloc[::-1]
        
        text = "【龙虎榜净买入TOP】\n"
        for _, row in top_buy.iterrows():
            name = row.get("名称", "未知")
            net_buy = row.get("净买入额", 0) / 1e8
            reason = row.get("上榜原因", "")
            text += f"- {name}：净买入 {net_buy:+.2f} 亿元（{reason}）\n"
        
        text += "\n【龙虎榜净卖出TOP】\n"
        for _, row in top_sell.iterrows():
            name = row.get("名称", "未知")
            net_sell = row.get("净买入额", 0) / 1e8
            text += f"- {name}：净卖出 {abs(net_sell):.2f} 亿元\n"
        return text
    except Exception as e:
        return f"龙虎榜数据获取失败：{str(e)}"

def get_margin_data():
    """获取融资融券市场情绪数据"""
    try:
        margin_df = ak.stock_margin_em()
        if margin_df.empty:
            return "暂无融资融券数据"
        latest = margin_df.iloc[-1]
        date = latest.get("交易日期", "")
        balance = latest.get("融资融券余额", 0) / 1e8
        net_buy = latest.get("融资净买入额", 0) / 1e8
        
        text = f"""【融资融券情绪】
- 统计日期：{date}
- 两融余额：{balance:.2f} 亿元
- 融资当日净买入：{net_buy:+.2f} 亿元
        """
        return text
    except Exception as e:
        return f"两融数据获取失败：{str(e)}"

def get_watchlist_data():
    """获取关注标的实时行情"""
    try:
        spot_df = ak.stock_zh_a_spot_em()
        if spot_df.empty:
            return "暂无个股行情数据"
        watch_df = spot_df[spot_df["代码"].isin(WATCHLIST_STOCKS)]
        if watch_df.empty:
            return "未找到关注标的，请检查6位股票代码"
        
        text = "【关注标的实时行情】\n"
        for _, row in watch_df.iterrows():
            name = row.get("名称", "未知")
            code = row.get("代码", "")
            price = row.get("最新价", 0)
            change = row.get("涨跌幅", 0)
            amount = row.get("成交额", 0) / 1e8
            text += f"- {name}（{code}）：{price:.2f}元，{change:+.2f}%，成交额{amount:.2f}亿\n"
        return text
    except Exception as e:
        return f"个股数据获取失败：{str(e)}"

def generate_analysis(index_data, news, market, industry, flow, lhb, margin, watchlist):
    """调用DeepSeek生成完整每日策略报告"""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1",
        timeout=60  # 增加超时时间，避免长时间卡住
    )
    
    prompt = f"""
你是资深A股策略分析师，请基于以下全维度数据撰写每日市场观察报告。
要求：
1. 分7个部分：大盘指数概览、核心政策要闻、行业盘面复盘、资金流向解读、龙虎榜与两融情绪、关注标的点评、明日操作提示
2. 语言简洁专业，总字数控制在1000字以内
3. 所有结论必须基于给出的真实数据，不编造信息，不构成投资建议

【大盘指数表现】
{index_data}

【财经要闻】
{news}

【北向资金】
{market}

【行业涨跌幅】
{industry}

【板块资金流向】
{flow}

【龙虎榜数据】
{lhb}

【融资融券】
{margin}

【关注标的】
{watchlist}
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def push_to_wechat(content):
    """Server酱推送至个人微信"""
    today = datetime.now().strftime("%Y年%m月%d日")
    title = f"A股每日策略观察 {today}"
    
    url = f"https://sctapi.ftqq.com/{SERVERCHAN_SENDKEY}.send"
    data = {"title": title, "desp": content}
    
    try:
        res = requests.post(url, data=data, timeout=10)
        result = res.json()
        if result.get("code") == 0:
            print("✅ 微信推送成功")
        else:
            print(f"❌ 推送失败：{result.get('message', '未知错误')}")
    except Exception as e:
        print(f"❌ 推送异常：{str(e)}")

def main():
    print("📥 开始采集全维度市场数据...")
    index_data = get_index_data()
    news = get_daily_news()
    market = get_market_data()
    industry = get_industry_data()
    flow = get_sector_flow()
    lhb = get_lhb_data()
    margin = get_margin_data()
    watchlist = get_watchlist_data()
    print("✅ 数据采集完成")
    
    print("🤖 正在生成策略分析报告...")
    report = generate_analysis(index_data, news, market, industry, flow, lhb, margin, watchlist)
    print("✅ 报告生成完成")
    
    print("📤 正在推送至微信...")
    push_to_wechat(report)
    print("🎉 今日任务全部完成")

if __name__ == "__main__":
    main()
