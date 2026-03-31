import os 

# find file size 
print(os.path.getsize("checkpoints/classifier.pth") / (1024 * 1024))  # Size in MB