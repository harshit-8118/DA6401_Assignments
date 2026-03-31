import os 

trainvaltxt = os.path.join('data', 'oxford-iiit-pet', 'annotations', 'trainval.txt')
testtxt = os.path.join('data', 'oxford-iiit-pet', 'annotations', 'test.txt')

imagepath = []
for f in [trainvaltxt, testtxt]:
    with open(f, 'r') as file:
        lines = file.readlines()
        imagepath.extend([line.split()[0] for line in lines])

# print(imagepath[:10], len(imagepath))
mask, bbox, img_cnt = 0, 0, 0

for img in imagepath:
    img_path = os.path.join('data', 'oxford-iiit-pet', 'annotations', 'trimaps', img + '.png')
    img_path2 = os.path.join('data', 'oxford-iiit-pet', 'images', img + '.jpg')
    img_path3 = os.path.join('data', 'oxford-iiit-pet', 'annotations', 'xmls', img + '.xml')
    if not os.path.exists(img_path):
        mask += 1
        print(f"File {img_path} does not exist.")

    if not os.path.exists(img_path2):
        img_cnt += 1
        print(f"File {img_path2} does not exist.")
    
    if not os.path.exists(img_path3):
        bbox += 1
        print(f"File {img_path3} does not exist.")

print(f"Total missing masks: {mask}")
print(f"Total missing images: {img_cnt}")
print(f"Total missing bboxes: {bbox}")
