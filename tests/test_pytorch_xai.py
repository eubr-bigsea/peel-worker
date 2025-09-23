from PIL import Image

from explainable_ai.pytorch_xai import ImageXAI
import torchvision.transforms as transforms
import torchvision.models as models


def test_pytorch_image_xai():
    # model expects 224x224 3-color image

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # standard ImageNet normalization
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    model = models.resnet18(weights='IMAGENET1K_V1')
    model = model.eval()

    img_dict = {
        'image_path': 'img_test/cat.jpg',
        'transformation': transform
    }

    img_xai = ImageXAI(model=model,
                       image=img_dict,
                       labels='img_test/imagenet_class_index.json',
                       transformation=transform_normalize)
    img_xai.set_all_attributes()

    img_xai.viz_grad_img()
    img_xai.viz_occ()
    img_xai.viz_layer()


