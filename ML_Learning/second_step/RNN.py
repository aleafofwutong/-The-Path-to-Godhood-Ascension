import torch 
import torch.nn as nn
from pathlib import Path
import sys 
import random
from torch.utils.data import Dataset, DataLoader
import re

# 初始化路径（保持原有配置）
ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))

# 你的 RNN 模型定义（保持不变）
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self,batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)

# -------------------------- 1. 内置中文古诗数据集（无需下载，无需联网） --------------------------
# 直接在代码中内置100首常见古诗（足够验证RNN训练流程）
built_in_poems = [
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "大漠孤烟直，长河落日圆。萧关逢候骑，都护在燕然。",
    "明月松间照，清泉石上流。竹喧归浣女，莲动下渔舟。",
    "青山遮不住，毕竟东流去。江晚正愁余，山深闻鹧鸪。",
    "人生自古谁无死，留取丹心照汗青。",
    "但愿人长久，千里共婵娟。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
    "葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。",
    "黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。",
    "慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。谁言寸草心，报得三春晖。",
    "小时不识月，呼作白玉盘。又疑瑶台镜，飞在青云端。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。",
    "朝发白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。",
    "相见时难别亦难，东风无力百花残。春蚕到死丝方尽，蜡炬成灰泪始干。",
    "锦瑟无端五十弦，一弦一柱思华年。庄生晓梦迷蝴蝶，望帝春心托杜鹃。",
    "昨夜星辰昨夜风，画楼西畔桂堂东。身无彩凤双飞翼，心有灵犀一点通。",
    "无题·相见时难别亦难，东风无力百花残。春蚕到死丝方尽，蜡炬成灰泪始干。",
    "山行：远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。",
    "清明：清明时节雨纷纷，路上行人欲断魂。借问酒家何处有？牧童遥指杏花村。",
    "泊船瓜洲：京口瓜洲一水间，钟山只隔数重山。春风又绿江南岸，明月何时照我还。",
    "元日：爆竹声中一岁除，春风送暖入屠苏。千门万户曈曈日，总把新桃换旧符。",
    "饮湖上初晴后雨：水光潋滟晴方好，山色空蒙雨亦奇。欲把西湖比西子，淡妆浓抹总相宜。",
    "题西林壁：横看成岭侧成峰，远近高低各不同。不识庐山真面目，只缘身在此山中。",
    "念奴娇·赤壁怀古：大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。",
    "水调歌头·明月几时有：明月几时有？把酒问青天。不知天上宫阙，今夕是何年。",
    "破阵子·为陈同甫赋壮词以寄之：醉里挑灯看剑，梦回吹角连营。八百里分麾下炙，五十弦翻塞外声。",
    "满江红·写怀：怒发冲冠，凭栏处、潇潇雨歇。抬望眼、仰天长啸，壮怀激烈。",
    "菩萨蛮·书江西造口壁：郁孤台下清江水，中间多少行人泪。西北望长安，可怜无数山。",
    "西江月·夜行黄沙道中：明月别枝惊鹊，清风半夜鸣蝉。稻花香里说丰年，听取蛙声一片。",
    "天净沙·秋思：枯藤老树昏鸦，小桥流水人家，古道西风瘦马。夕阳西下，断肠人在天涯。",
    "山坡羊·潼关怀古：峰峦如聚，波涛如怒，山河表里潼关路。望西都，意踌躇。",
    "己亥杂诗·其五：浩荡离愁白日斜，吟鞭东指即天涯。落红不是无情物，化作春泥更护花。",
    "游山西村：莫笑农家腊酒浑，丰年留客足鸡豚。山重水复疑无路，柳暗花明又一村。",
    "过零丁洋：辛苦遭逢起一经，干戈寥落四周星。山河破碎风飘絮，身世浮沉雨打萍。",
    "石灰吟：千锤万凿出深山，烈火焚烧若等闲。粉骨碎身浑不怕，要留清白在人间。",
    "竹石：咬定青山不放松，立根原在破岩中。千磨万击还坚劲，任尔东西南北风。",
    "村居：草长莺飞二月天，拂堤杨柳醉春烟。儿童散学归来早，忙趁东风放纸鸢。",
    "所见：牧童骑黄牛，歌声振林樾。意欲捕鸣蝉，忽然闭口立。",
    "小池：泉眼无声惜细流，树阴照水爱晴柔。小荷才露尖尖角，早有蜻蜓立上头。",
    "晓出净慈寺送林子方：毕竟西湖六月中，风光不与四时同。接天莲叶无穷碧，映日荷花别样红。",
    "四时田园杂兴·其二十五：梅子金黄杏子肥，麦花雪白菜花稀。日长篱落无人过，惟有蜻蜓蛱蝶飞。",
    "乡村四月：绿遍山原白满川，子规声里雨如烟。乡村四月闲人少，才了蚕桑又插田。",
    "渔歌子·西塞山前白鹭飞：西塞山前白鹭飞，桃花流水鳜鱼肥。青箬笠，绿蓑衣，斜风细雨不须归。",
    "浪淘沙·九曲黄河万里沙：九曲黄河万里沙，浪淘风簸自天涯。如今直上银河去，同到牵牛织女家。",
    "忆江南·江南好：江南好，风景旧曾谙。日出江花红胜火，春来江水绿如蓝。能不忆江南？",
    "长相思·山一程：山一程，水一程，身向榆关那畔行，夜深千帐灯。风一更，雪一更，聒碎乡心梦不成，故园无此声。",
    "泊秦淮：烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。",
    "夜雨寄北：君问归期未有期，巴山夜雨涨秋池。何当共剪西窗烛，却话巴山夜雨时。",
    "无题·昨夜星辰昨夜风：昨夜星辰昨夜风，画楼西畔桂堂东。身无彩凤双飞翼，心有灵犀一点通。",
    "锦瑟：锦瑟无端五十弦，一弦一柱思华年。庄生晓梦迷蝴蝶，望帝春心托杜鹃。",
    "登高：风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。",
    "茅屋为秋风所破歌：八月秋高风怒号，卷我屋上三重茅。茅飞渡江洒江郊，高者挂罥长林梢。",
    "卖炭翁：卖炭翁，伐薪烧炭南山中。满面尘灰烟火色，两鬓苍苍十指黑。",
    "琵琶行：浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。",
    "长恨歌：汉皇重色思倾国，御宇多年求不得。杨家有女初长成，养在深闺人未识。",
    "春江花月夜：春江潮水连海平，海上明月共潮生。滟滟随波千万里，何处春江无月明！",
    "月下独酌：花间一壶酒，独酌无相亲。举杯邀明月，对影成三人。",
    "将进酒：君不见，黄河之水天上来，奔流到海不复回。君不见，高堂明镜悲白发，朝如青丝暮成雪。",
    "蜀道难：噫吁嚱，危乎高哉！蜀道之难，难于上青天！蚕丛及鱼凫，开国何茫然！",
    "梦游天姥吟留别：海客谈瀛洲，烟涛微茫信难求；越人语天姥，云霞明灭或可睹。",
    "行路难·其一：金樽清酒斗十千，玉盘珍羞直万钱。停杯投箸不能食，拔剑四顾心茫然。",
    "早发白帝城：朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "望庐山瀑布：日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "静夜思：床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "赠汪伦：李白乘舟将欲行，忽闻岸上踏歌声。桃花潭水深千尺，不及汪伦送我情。",
    "黄鹤楼送孟浩然之广陵：故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
    "望天门山：天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。",
    "独坐敬亭山：众鸟高飞尽，孤云独去闲。相看两不厌，只有敬亭山。",
    "秋浦歌·其十五：白发三千丈，缘愁似个长。不知明镜里，何处得秋霜。",
    "月下独酌·其四：穷愁千万端，美酒三百杯。愁多酒虽少，酒倾愁不来。",
    "宣州谢朓楼饯别校书叔云：弃我去者，昨日之日不可留；乱我心者，今日之日多烦忧。",
    "把酒问月·故人贾淳令予问之：青天有月来几时？我今停杯一问之。人攀明月不可得，月行却与人相随。",
    "峨眉山月歌：峨眉山月半轮秋，影入平羌江水流。夜发清溪向三峡，思君不见下渝州。",
    "春夜洛城闻笛：谁家玉笛暗飞声，散入春风满洛城。此夜曲中闻折柳，何人不起故园情。",
    "渡荆门送别：渡远荆门外，来从楚国游。山随平野尽，江入大荒流。",
    "送友人：青山横北郭，白水绕东城。此地一为别，孤蓬万里征。",
    "听蜀僧濬弹琴：蜀僧抱绿绮，西下峨眉峰。为我一挥手，如听万壑松。",
    "夜宿山寺：危楼高百尺，手可摘星辰。不敢高声语，恐惊天上人。",
    "独坐敬亭山：众鸟高飞尽，孤云独去闲。相看两不厌，只有敬亭山。",
    "秋浦歌·其十四：炉火照天地，红星乱紫烟。赧郎明月夜，歌曲动寒川。",
    "怨情：美人卷珠帘，深坐颦蛾眉。但见泪痕湿，不知心恨谁。",
    "玉阶怨：玉阶生白露，夜久侵罗袜。却下水晶帘，玲珑望秋月。",
    "问刘十九：绿蚁新醅酒，红泥小火炉。晚来天欲雪，能饮一杯无？",
    "江雪：千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "寻隐者不遇：松下问童子，言师采药去。只在此山中，云深不知处。",
    "枫桥夜泊：月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "渔歌子：西塞山前白鹭飞，桃花流水鳜鱼肥。青箬笠，绿蓑衣，斜风细雨不须归。",
    "寒食：春城无处不飞花，寒食东风御柳斜。日暮汉宫传蜡烛，轻烟散入五侯家。",
    "绝句·两个黄鹂鸣翠柳：两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "绝句·迟日江山丽：迟日江山丽，春风花草香。泥融飞燕子，沙暖睡鸳鸯。",
    "江畔独步寻花·其六：黄四娘家花满蹊，千朵万朵压枝低。留连戏蝶时时舞，自在娇莺恰恰啼。",
    "春望：国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。",
    "登高：风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。"
]

# 数据清洗：合并所有古诗为纯文本，过滤无效字符
all_text = "\n".join([poem.strip() for poem in built_in_poems if poem.strip()])
all_text = re.sub(r"[^\u4e00-\u9fa5\n，。！？；：、]", "", all_text)  # 只保留中文和常用标点

# 构建字符映射表
chars = sorted(list(set(all_text)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

print(f"词汇表大小（字符数）：{vocab_size}")
print(f"前10个字符：{chars[:10]}")

# -------------------------- 2. 构建训练数据集（保持不变） --------------------------
seq_len = 30
step = 1

def create_train_data(all_text, seq_len, step):
    x_data = []
    y_data = []
    text_idx = [char2idx[c] for c in all_text]
    
    for i in range(0, len(text_idx) - seq_len, step):
        x_seq = text_idx[i:i+seq_len]
        y_seq = text_idx[i+1:i+seq_len+1]
        x_data.append(x_seq)
        y_data.append(y_seq)
    
    return torch.tensor(x_data, dtype=torch.long), torch.tensor(y_data, dtype=torch.long)

x_train, y_train = create_train_data(all_text, seq_len, step)

class PoemDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 适配样本数的batch_size
batch_size = 32 if len(x_train) >= 32 else len(x_train) if len(x_train) > 0 else 1
train_dataset = PoemDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------- 3. 模型初始化与训练配置（保持不变） --------------------------
hidden_size = 256
input_size = vocab_size
output_size = vocab_size
learning_rate = 0.001
epochs = 150  # 内置数据集量小，15轮足够收敛

model = RNN(input_size, hidden_size, output_size)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"使用设备：{device}")

# -------------------------- 4. 模型训练（保持不变） --------------------------
print("开始训练...")
for epoch in range(epochs):
    total_loss = 0.0
    model.train()
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_loss = 0.0
        
        for i in range(seq_len):
            input_char = torch.nn.functional.one_hot(batch_x[:, i], num_classes=vocab_size).float()
            hidden = model.initHidden(batch_size=batch_x.shape[0]).to(device)
            output, hidden = model(input_char, hidden)
            loss = criterion(output, batch_y[:, i])
            batch_loss += loss
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), ROOT / "poem_rnn_model.pth")
print("模型保存完成：", ROOT / "poem_rnn_model.pth")

# -------------------------- 5. 古诗生成（保持不变） --------------------------
def generate_poem(model, start_str, generate_len=100, temperature=0.7):
    model.eval()
    hidden = model.initHidden().to(device)
    generated = start_str
    
    # 初始化隐藏状态
    for c in start_str:
        if c not in char2idx:
            c = chars[0]
        input_char = torch.nn.functional.one_hot(torch.tensor([char2idx[c]]), num_classes=vocab_size).float().to(device)
        output, hidden = model(input_char, hidden)
    
    # 生成后续字符
    current_char = start_str[-1]
    for _ in range(generate_len):
        if current_char not in char2idx:
            current_char = chars[0]
        input_char = torch.nn.functional.one_hot(torch.tensor([char2idx[current_char]]), num_classes=vocab_size).float().to(device)
        output, hidden = model(input_char, hidden)
        
        output = output / temperature
        probs = torch.softmax(output, dim=1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = idx2char[next_idx]
        
        generated += next_char
        current_char = next_char
    
    return generated.replace("\n", "\n")

# 测试生成
start_str = "春眠不觉晓"
generated_poem = generate_poem(model, start_str, generate_len=80, temperature=0.6)
print("\n生成的古诗：")
print("="*50)
print(generated_poem)