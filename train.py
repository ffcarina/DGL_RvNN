import collections
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.utils.data import DataLoader
from display_tool import table
import dgl
from dgl.data.tree import SST, SSTBatch
from RvNN import RvNN

Epoch = 50
emb_LR = 1e-3
LR = 1e-2  
L2_reg = 0.001
batch_size = 128
emb_dim = 300
dropout = 0.5  # 0.1 too small
seed = 50

# SSTBatch = collections.namedtuple('SSTBatch', ['graph', 'mask', 'wordid', 'label'])
def batcher():
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(graph=batch_trees,
                        mask=batch_trees.ndata['mask'].cuda(),
                        wordid=batch_trees.ndata['x'].cuda(),
                        label=batch_trees.ndata['y'].cuda())
    return batcher_dev


def main():
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    bestAll_epoch = -1
    bestRoot_epoch = -1
    bestRoot_acc = 0
    bestAll_acc = 0


    trainset = SST()  # default mode='train'
    vocab = trainset.vocab  # inclueding train,dev,test
    word_to_index = {word: id for word, id in vocab.items()} # inverted vocabulary dict: word -> id

    train_loader = DataLoader(dataset=trainset,
                              batch_size=batch_size, 
                              collate_fn=batcher(),
                              shuffle=True,
                              num_workers=0)

    testset = SST(mode='test')
    test_loader = DataLoader(dataset=testset,
                             batch_size=100, collate_fn=batcher(), shuffle=False, num_workers=0)

    model = RvNN(word_to_index,
                trainset.num_vocabs,
                emb_dim,
                trainset.num_classes,
                dropout).cuda()  
    print(model)
    # embedding和其他的参数变量分开，为了设置不同的学习率
    params_ex_emb =[x for x in list(model.parameters()) if x.requires_grad and x.size(0)!=trainset.num_vocabs]

    params_emb = list(model.embedding.parameters())

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)   # 使用均匀分布初始化参数

    optimizer = optim.Adagrad([
        {'params':params_ex_emb, 'lr':LR, 'weight_decay':L2_reg}, 
        {'params':params_emb, 'lr':emb_LR}])   
    
    pt = table(["epoch", "Test Acc", "Root Acc", "Epoch Time"])
    t_epoch = time.time()  # start time
    start = t_epoch
    for epoch in range(Epoch):   # epochs
        model.train()
        for step, batch in enumerate(train_loader):
            g = batch.graph
            # g.set_n_initializer(dgl.init.zero_initializer)
            n = g.number_of_nodes()
            logits = model(batch)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label, reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = time.time()
        # test
        accs = []
        root_accs = []
        model.eval()

        for step, batch in enumerate(test_loader):
            g = batch.graph
            g.set_n_initializer(dgl.init.zero_initializer)
            n = g.number_of_nodes()
            # 禁止梯度计算
            with    torch.no_grad():
                
                logits = model(batch)  # (n, 5)

            pred =  torch.argmax(logits, 1)  # (n, 1)
            acc = torch.sum(torch.eq(batch.label, pred)).item()
            accs.append([acc, len(batch.label)])
            root_ids = [i for i in range(n) if batch.graph.out_degree(i) == 0]
            # root_acc = torch.sum(batch.label.cpu().data.numpy()[root_ids] == pred.cpu().data.numpy()[root_ids])
            root_acc =  torch.sum(batch.label.data[root_ids] == pred.data[root_ids]).item()
            root_accs.append([root_acc, len(root_ids)])

        acc = 1.0 * np.sum([x[0] for x in accs]) / np.sum([x[1] for x in accs])
        root_acc = 1.0 * np.sum([x[0] for x in root_accs]) / np.sum([x[1] for x in root_accs])

        if acc > bestAll_acc:
            bestAll_acc = acc
            bestAll_epoch = epoch
        if root_acc > bestRoot_acc:
            bestRoot_acc = root_acc
            bestRoot_epoch = epoch

        pt.row([epoch, acc, root_acc, end - start])
        start = end

    # summary, including total time of training
    print("BestAll_epoch_test: {}   BestAll_acc_test: {:.4f}".format(bestAll_epoch, bestAll_acc))
    print("BestRoot_epoch_test: {}   BestRoot_acc_test: {:.4f}".format(bestRoot_epoch, bestRoot_acc))
    print("Total time:", time.time() - t_epoch)


if __name__ == '__main__':
    main()
    
