# 微信聊天记录分析

本篇主要介绍如何基于 WeChatMsg 和 Python 进行微信聊天记录的分析。

## 1. 数据准备

仓库地址：[WeChatMsg](https://github.com/LC044/WeChatMsg)

此开源软件可以解析电脑版微信的数据库，并提供基本的数据分析

### 1.1 安装 WeChatMsg

[如何安装 WeChatMsg](https://github.com/LC044/WeChatMsg/blob/master/doc/%E5%BC%80%E5%8F%91%E8%80%85%E6%89%8B%E5%86%8C.md)

参考 WeChatMsg 的开发者手册对仓库进行 clone，并运行 python 程序本体

```shell
# Python>=3.10 仅支持3.10、3.11、3.12,请勿使用其他Python版本
git clone https://github.com/LC044/WeChatMsg
# 网络不好推荐用Gitee
# git clone https://gitee.com/lc044/WeChatMsg.git
cd WeChatMsg
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行程序
python main.py
```

### 1.2 使用 WeChatMsg

在运行该程序时，需要保证电脑版微信在线。同时，由于软件提取的微信消息记录基于电脑中存储的记录，因此建议在提取前做好移动设备的消息迁移，保证消息记录的完整性。

按照该程序指引进行数据库提取，提取数据库之后可以基于数据库进行进一步的分析

同时也可以进入“聊天”子页面，对不同的联系人查看简单的年度报告分析

## 2. 利用 python 进行自定义数据分析

WeChatMsg 提供的报告分析功能较为有限，但提供了导出 csv 等格式，可以利用这个功能对消息记录进行进一步的分析。

部分分析功能的参考代码：[WeChatAnalysis](https://github.com/Erix025/WeChatAnalysis)

### 2.0 读取 csv 文件并获取所有消息记录

1. 定义一个 Message 类

```python
class Msg:
    time: datetime
    sender: str
    content: str
    type: int
    icon: str
    receiver: str

    def __init__(self, item):
        self.time = datetime.strptime(item[8], "%Y-%m-%d %H:%M:%S")
        self.sender = item[10]
        if item[4] == "1":
            self.sender = "Eric025"
        self.content = item[7]
        self.type = int(item[2])
        if self.type != 47:
            self.icon = None
        else:
            emoji = icons.parser_xml(self.content)
            md5 = emoji.get("md5")
            self.icon = md5

    def __str__(self):
        return f"{self.time} {self.sender}: {self.content}"
```

2. 进行 csv 文件的读取

```python
messages = []

with open("filename", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in tqdm.tqdm(reader):
        # skip headline
        if row[4] != "1" and row[4] != "0":
            continue
        messages.append(Msg(row))
```

### 2.1 基础的消息统计

例如：根据信息的 Type 字段进行分类、级数；或者根据消息的发送时间进行统计

这部分代码比较简单，省略不写。需要参考具体实例详见代码仓库。

### 2.2 群聊各成员活跃天数统计

活跃天数统计：

```python
# 统计每一天的消息量
def activate_days(messages):
    counter = {}
    for msg in messages:
        date = msg.time.strftime("%Y-%m-%d")
        if date not in counter:
            counter[date] = 0
        counter[date] += 1
    return counter

# 统计每个人的活跃日期，并与指定时间段的日期匹配，返回缺勤的日期
def activate_days_rank(messages):
    # 获得群聊名单
    # 此处调用函数，内部是非常naive的直接遍历所有message统计所有发送者
    senders = get_sender_list(messages)
    counter = {}
    # 获取指定日期范围内的日期列表，用于展现缺勤日期
    start_time = datetime(2023, 2, 15)
    end_time = datetime(2024, 2, 15)
    date_list = [
        (start_time + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end_time - start_time).days)
    ]
    for sender in senders:
        # 获取指定sender的消息列表，并进行统计
        sender_messages = [msg for msg in messages if msg.sender == sender]
        days = activate_days(sender_messages)
        counter[sender] = [key for key in date_list if key not in days.keys()]
    counter = dict(sorted(counter.items(), key=lambda x: len(x[1])))
    print(counter)
```

### 2.3 最爱表情统计

通过统计消息中出现最多次数的表情，展示 top20 的表情列表。下载获取表情文件的代码参考了 WeChatMsg 中的实现。

关于获取表情的函数在`icons.py`文件中，可从参考代码的仓库中获取。

```python
import icons

def top_icons(messages):
    icon_counts = {}
    for msg in messages:
        if msg.type == 47:
            icon = msg.icon
            if icon:
                if icon not in icon_counts:
                    icon_counts[icon] = 0
                icon_counts[icon] += 1
    # print most used icon
    # 获取排名前20的表情
    icon_counts = dict(sorted(icon_counts.items(), key=lambda x: x[1], reverse=True))
    icon_counts = dict(list(icon_counts.items())[:20])
    most_used_xml_strings = {}
    for msg in messages:
        if msg.icon in icon_counts.keys():
            if msg.icon not in most_used_xml_strings.keys():
                most_used_xml_strings[msg.icon] = msg.content
        if len(most_used_xml_strings) == 20:
            break
    image_paths = []
    for icon in icon_counts.keys():
        image_paths.append(icons.get_emoji(most_used_xml_strings[icon]))
    print(icon_counts)

    # 创建一个新的图形
    fig = plt.figure(figsize=(20, 20))
    print(image_paths)
    # 按顺序显示每一张图片
    for i, icon in enumerate(icon_counts.keys()):
        # 读取图片
        image_path = image_paths[i]
        try:
            img = mpimg.imread(image_path)
        except:
            continue
        # 创建一个子图
        ax = fig.add_subplot(5, 4, i + 1)

        # 显示图片
        ax.imshow(img)

        # 设置子图的标题
        ax.set_title(f"{icon_counts[icon]}次")

        ax.axis("off")
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # 显示图形
    plt.show()
```

### 2.4 词频统计

利用 jieba 库进行分词，并统计高频词汇

```python
def word_freq(messages):
    # 统计词频
    import jieba
    from collections import Counter

    contents = [msg.content for msg in messages if msg.type == 1]
    text = "".join(contents)
    words = jieba.cut_for_search(text)
    # 此处添加了对单字的过滤，可以改为一些自定义规则
    counter = Counter([word for word in words if len(word) > 1])
    print(counter.most_common(100))

```

### 2.5 发言情感统计

利用 SnowNLP 库进行消息情感分析统计，并返回最正面/负面的消息。
注：实际使用发现该库的表现在抽象中文互联网词汇的大海中表现并不好，并且容易将长信息视为极度 positive/negative，建议根据结果添加进一步的自定义规则，并请慎重对待分析结果。

```py
def sentiment_analysis(messages):
    # 添加了自定义筛选规则
    # 限制信息长度，以免出现长信息的误判
    # 过滤掉"捂脸"，该词的出现会使信息被视为极度negative的信息
    contents = [
        msg.content
        for msg in messages
        if msg.type == 1 and len(msg.content) < 50 and "捂脸" not in msg.content
    ]
    if len(contents) < 5:
        return
    sentiments = {}
    for msg in tqdm.tqdm(contents):
        sentiments[msg] = SnowNLP(msg).sentiments
    neg_count = len([s for s in sentiments.values() if s < 0.3])
    pos_count = len([s for s in sentiments.values() if s > 0.7])
    print(f"positive: {pos_count}")
    print(f"negative: {neg_count}")
    print(f"average:{sum(sentiments.values())/len(sentiments)}")
    most_5_positive = sorted(sentiments.items(), key=lambda x: x[1], reverse=True)[:5]
    most_5_negative = sorted(sentiments.items(), key=lambda x: x[1])[:5]
    print(most_5_positive)
    print(most_5_negative)
```

其他更多有意思的统计有待补充，更多参考代码可以参考 github 仓库：
[WeChatAnalysis](https://github.com/Erix025/WeChatAnalysis)
