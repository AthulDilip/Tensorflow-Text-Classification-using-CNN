import text_classifier

x_text = ["ultimately , it ponders the reasons we need stories so much . ", "Can be better.", "It was okay", "the lively appeal of the last kiss lies in the ease with which it integrates thoughtfulness and pasta-fagioli comedy . "]
checkpoint_dir = "runs/1495097016/checkpoints/"

result = text_classifier.classify(checkpoint_dir, x_text)
print(result)