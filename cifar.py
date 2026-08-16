from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision
from torchvision import transforms

cifar = Flask(__name__)
# import model.py
#from model import dog_cat
#model = dog_cat()
from model import CNN

model = CNN()

model.load_state_dict(
    torch.load('cifar10_cnn.pth', map_location='cpu')
    )

model.eval()
# load IMAGE
#model.load_state_dict(torch.load('cifar.path', map_location='cpu'))
#transform the shape

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

@cifar.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        prediction=None
        #image from user
        image=request.files['image']
        img=Image.open(image).convert('RGB')
        img=transform(img)
        img= img.unsqueeze(dim=0)
        classes=['airplane', 'automobile', 
              'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
              'ship', 'truck']
        with torch.no_grad():
            output=model(img)
            y_pred=output.argmax(dim=1).item()
            # Convert number to class name
        prediction = classes[y_pred]

        return render_template(
            'cifar.html',
            prediction=prediction
        )

    return render_template('cifar.html')


if __name__ == '__main__':
    cifar.run(debug=True)















