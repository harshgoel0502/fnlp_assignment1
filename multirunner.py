# import subprocess
#
# if __name__ == '__main__':
#     with open('results_wv.txt', 'w') as f:
#         for epoch in range(3,8):
#             for batch in range(5,30,5):
#                 for lr in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
#                     result = subprocess.run(['python3', 'run_classifier.py', '--model', 'LR', '--tokenizer', 'NONE', '--feats', 'WV', '--epochs', str(epoch), '--batch_size', str(batch), '--learning_rate', str(lr)], stdout=subprocess.PIPE).stdout.decode('utf-8')
#         # result = subprocess.run(['python3', 'run_classifier.py', '--model', 'LR', '--tokenizer', 'NGRAM', '--ngrams', '2', '--feats', 'COUNTER', '--epochs', '4', '--batch_size', '5', '--learning_rate', '0.5'], stdout=subprocess.PIPE).stdout.decode('utf-8')
#                     f.write(result)
#                     f.write('\n---------------------------------------\n\n\n')
#

if __name__ == '__main__':
    with open('results_wv.txt', 'r') as f:
        count = 0
        best_acc_hp = ""
        best_acc = 0.0
        best_dev_hp = ""
        best_dev = 0.0
        best_test_hp = ""
        best_test = 0.0
        name = ""
        for line in f:
            if "Namespace" in line:
                name = line
            if "Accuracy:" in line:
                if count == 0:
                    if float(line.split("=")[1].strip()) > best_acc:
                        best_acc = float(line.split("=")[1].strip())
                        best_acc_hp = name
                    count += 1
                elif count == 1:
                    if float(line.split("=")[1].strip()) > best_dev:
                        best_dev = float(line.split("=")[1].strip())
                        best_dev_hp = name
                    count += 1
                elif count == 2:
                    if float(line.split("=")[1].strip()) > best_test:
                        best_test = float(line.split("=")[1].strip())
                        best_test_hp = name
                    count = 0
        print(best_acc_hp)
        print(best_acc)
        print(best_dev_hp)
        print(best_dev)
        print(best_test_hp)
        print(best_test)