from data_loader import DataLoader
from base_seq2seq import Base, Attention, Encoder, Decoder
from torch import nn, optim
import torch, time, random


def main():
    english_data, german_data = get_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(english_data, german_data, device)
    checkpoint = torch.load("checkpoint_9_300.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    params = {}
    params['batch_size'] = 40
    params['epochs'] = 10
    params['learning_rate'] = 0.001

    test(english_data['dev'], german_data['dev'], model, params)


def test(eng_dev, de_dev, net, params):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    
    num_examples, eng_len = eng_dev.size() 

    batches = [(start, start + params['batch_size']) for start in\
               range(0, num_examples, params['batch_size'])]
    
    net.eval()
    all_preds = []
    with torch.no_grad():
        for epoch in range(params['epochs']):
            ep_loss = 0
            
            start_time = time.time()
            random.shuffle(batches)
            
            # for each batch, calculate loss and optimize model parameters            
            len_batch = len(batches)

            for b_idx, (start, end) in enumerate(batches):
                de_src = de_dev[start:end]
                eng_trg = eng_dev[start:end]
                preds = net(de_src, eng_trg, 0)
                
                # q1.1: explain the below line!
                preds = preds[1:].view(-1, preds.shape[-1])
                eng_target = eng_trg[1:].view(-1)
                print(preds)
                all_preds.append(preds)
                loss = criterion(preds, eng_target)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ep_loss += loss
            
            print('epoch: {0}, loss: {1}, time: {2}'.format(epoch, ep_loss, time.time()-start_time))
    

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            print(output)
            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


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
    english_data['dev'] = torch.LongTensor(english_data['dev']).cuda()
    german_data['dev'] = torch.LongTensor(german_data['dev']).cuda()
    return english_data, german_data