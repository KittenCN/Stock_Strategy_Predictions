import random
from common import *
import pickle

# 定义超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.01
LOOKBACK = SEQ_LEN
FORECAST = 1
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_datas = []
train_codes = []
test_codes = []
ts_codes =[]
train_list = []
test_list = []
total_length = 0
total_test_length = 0

csv_files = glob.glob(daily_path+"/*.csv")
for csv_file in csv_files:
    ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])

if len(ts_codes) > 1:
    if os.path.exists("test_codes.txt"):
        with open("test_codes.txt", 'r') as f:
            test_codes = f.read().splitlines()
        train_codes = list(set(ts_codes) - set(test_codes))
    else:
        train_codes = random.sample(ts_codes, int(TRAIN_WEIGHT*len(ts_codes)))
        test_codes = list(set(ts_codes) - set(train_codes))
        with open("test_codes.txt", 'w') as f:
            for test_code in test_codes:
                f.write(test_code + "\n")
else:
    train_codes = ts_codes
    test_codes = ts_codes
random.shuffle(ts_codes)
random.shuffle(train_codes)
random.shuffle(test_codes)

with open(train_pkl_path, 'rb') as f:
    _data_queue = dill.load(f)
    while _data_queue.empty() == False:
        try:
            _datas.append(_data_queue.get(timeout=30))
        except queue.Empty:
            break
    random.shuffle(_datas)
    init_bar = tqdm(total=len(_datas), ncols=TQDM_NCOLS)
    for _data in _datas:
        init_bar.update(1)
        _data = _data.fillna(_data.median(numeric_only=True))
        if _data.empty:
            continue
        _ts_code = str(_data['ts_code'][0]).zfill(6)
        _ts_code = _ts_code.zfill(6)
        if _ts_code in train_codes:
            train_list.append(_data)
            total_length += _data.shape[0] - SEQ_LEN
        if _ts_code in test_codes:
            test_list.append(_data)
            total_test_length += _data.shape[0] - SEQ_LEN
        if _ts_code not in train_codes and _ts_code not in test_codes:
            print("Error: %s not in train or test"%_ts_code)
            continue
        if _ts_code in train_codes and _ts_code in test_codes:
            print("Error: %s in train and test"%_ts_code)
            continue
    init_bar.close()
codes_len = len(train_codes)
print("total codes: %d, total length: %d"%(codes_len, total_length))
print("total test codes: %d, total test length: %d"%(len(test_codes), total_test_length))

# 创建数据加载器
train_dataloader = StockDataset(train_list, lookback=LOOKBACK, forecast=FORECAST)  # 你需要定义你的训练数据加载器
val_dataloader = StockDataset(test_list, lookback=LOOKBACK, forecast=FORECAST)  # 你需要定义你的验证数据加载器

# 初始化模型 ntoken, ninp, nhead, nhid, nlayers, dropout=0.5
model = newTransformerModel(ntoken=INPUT_DIMENSION, ninp=D_MODEL, nhead=NHEAD, nhid=HIDDEN_DIMENSION, nlayers=NUM_LAYERS, dropout=0.5).to(DEVICE)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 开始训练
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    for batch in train_dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        total_loss = 0
        correct_predictions = 0
        for batch in val_dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # 计算正确预测的数量
            predicted = torch.sigmoid(outputs) > 0.5
            correct = (predicted == targets).sum().item()
            correct_predictions += correct

        # 打印每轮的损失和准确率
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(val_dataloader)}, Accuracy: {correct_predictions/len(val_dataloader.dataset)}')