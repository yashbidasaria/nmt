from data_loader import DataLoader
from base_seq2seq import Base, Attention, Encoder, Decoder
from torch import nn, optim
import torch, time, random

def main():
    english_data, german_data = get_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(english_data, german_data, device)
    params = {}
    params['batch_size'] = 40
    params['epochs'] = 10
    params['learning_rate'] = 0.001

    train(english_data['train'], english_data['dev'],
          german_data['train'], german_data['dev'], model, params)

def create_model(english_data, german_data, device):    
    INPUT_DIM = len(german_data['idx2word'])
    OUTPUT_DIM = len(english_data['idx2word'])
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Base(enc, dec, device).to(device)
    return model


def get_data():
    german_file = "german200.pickle"
    english_file = "english200.pickle"
    torch.cuda.empty_cache()
    file_loader = DataLoader(german_file, english_file)
    german_data = file_loader.get_german()
    english_data = file_loader.get_english()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    english_data['train'] = torch.LongTensor(english_data['train']).cuda()
    english_data['dev'] = torch.LongTensor(english_data['dev']).cuda()
    german_data['train'] = torch.LongTensor(german_data['train']).cuda()
    german_data['dev'] = torch.LongTensor(german_data['dev']).cuda()
    return english_data, german_data

def train(eng_train, eng_dev, de_train, de_dev, net, params):
  
    # padding is 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    
    num_examples, eng_len = eng_train.size()    
    batches = [(start, start + params['batch_size']) for start in\
               range(0, num_examples, params['batch_size'])]
    
    
    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        random.shuffle(batches)
        
        # for each batch, calculate loss and optimize model parameters            
        accumulation_steps = 5
        len_batch = len(batches)
        for b_idx, (start, end) in enumerate(batches):
            if b_idx % 15 == 0:
                print("batch: ", str(b_idx), " total: ", str(len_batch))
            de_src = de_train[start:end]
            eng_trg = eng_train[start:end]
            preds = net(de_src, eng_trg)
            
            # q1.1: explain the below line!
            preds = preds[1:].view(-1, preds.shape[-1])
            eng_target = eng_trg[1:].view(-1)
            loss = criterion(preds, eng_target)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ep_loss += loss
        
        checkpoint = {}
        checkpoint['state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        file_name = 'checkpoints/checkpoint_{0}_300.pth'.format(epoch)
        torch.save(checkpoint, file_name)
        print('epoch: {0}, loss: {1}, time: {2}'.format(epoch, ep_loss, time.time()-start_time))


if __name__ == "__main__":
    main()
