from PIL import Image
im=Image.open("KH/KH_2.jpg")
images=[]
for i in range(4,112,2):
    images.append(Image.open('KH/KH_'+str(i)+'.jpg'))

im.save('KH1.gif', save_all=True, append_images=images,loop=1,duration=0.1)
