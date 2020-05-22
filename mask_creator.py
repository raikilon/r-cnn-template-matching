import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


class AugmentedDataset():
    def __init__(self, root):
        self.root = root

        # self.imgs_backgrounds = list(sorted(os.listdir(os.path.join(root, 'backgrounds'))))
        self.imgs_backgrounds = list(
            sorted([f for f in os.listdir(os.path.join(root, 'backgrounds')) if not f.startswith('.')]))
        self.imgs_templates = list(
            sorted([f for f in os.listdir(os.path.join(root, 'templates')) if not f.startswith('.')]))

        self.template_masks = self.create_template_masks(self.imgs_templates)

        # number of max copies of the same template in one augmented image
        self.max_templates = 3
        # maximum relation between background to template
        # the larger this value, the smaller is the maximum template size relative to its background
        self.max_temp_back_rel = 4
        # min scale of template when scaling the image down

        # the larger this value, the larger is the minimum template size relative to its background
        self.min_augm_scale = 0.5
        # max rotation angle in the augmentation
        self.max_augm_rot = 40.0

    def create_template_masks(self, imgs_templates):
        template_masks = []
        for template_name in imgs_templates:
            template = cv2.imread(os.path.join(self.root, 'templates', template_name), cv2.IMREAD_UNCHANGED)
            alpha_channel = template[:, :, 3]
            _, template_mask = cv2.threshold(alpha_channel, 254, 1, cv2.THRESH_BINARY)
            template_mask = template_mask.reshape((alpha_channel.shape[0], alpha_channel.shape[1]))
            template_masks.append(template_mask)
        return template_masks

    def get_random_background(self):
        idx = np.random.randint(0, len(self.imgs_backgrounds))
        return self.imgs_backgrounds[idx]

    def augment_templates(self, template, template_mask, amount):
        augmented_templates = []
        augmented_template_masks = []
        for i in range(amount):
            scale = np.random.uniform(self.min_augm_scale, 1.0)
            template = cv2.resize(template, dsize=(0, 0), fx=scale, fy=scale)
            template_mask = cv2.resize(template_mask, dsize=(0, 0), fx=scale, fy=scale)
            angle = np.random.uniform(-self.max_augm_rot, self.max_augm_rot)
            template = rotate(template, angle)
            template_mask = rotate(template_mask, angle)

            augmented_templates.append(template)
            augmented_template_masks.append(template_mask)

        return augmented_templates, augmented_template_masks

    def stitch_templates_to_background(self, background, templates, template_masks, temp_count):
        back_height = np.shape(background)[0]
        back_width = np.shape(background)[1]
        temp_cumsum = np.cumsum(temp_count)
        count_temp_cumsum = np.zeros(len(temp_count))

        mask_array = [np.zeros_like(background[:, :, 0]) for _ in range(len(templates))]

        for k in range(len(templates)):
            # if len(templates) > 0:

            rand_idx_array = np.random.permutation(len(templates[k]))
            print(np.shape(templates))
            for idx in rand_idx_array:
                template = templates[k][idx]
                template_mask = template_masks[k][idx]
                temp_height = np.shape(template)[0]
                temp_width = np.shape(template)[1]
                y_coord = np.random.randint(0, back_height - temp_height)
                x_coord = np.random.randint(0, back_width - temp_width)
                y1, y2 = y_coord, y_coord + temp_height
                x1, x2 = x_coord, x_coord + temp_width

                i = 0
                while i < len(templates[k]):
                    if idx <= temp_cumsum[i]:
                        count_temp_cumsum[i] = count_temp_cumsum[i] + 1

                        #mask_array[k][y1:y2, x1:x2] = template_mask * count_temp_cumsum[i]

                        for y in range(temp_height):
                            for x in range(temp_width):
                                if template_mask[y,x] > 0:
                                    mask_array[k][y1+y, x1+x] = template_mask[y,x] * count_temp_cumsum[i]
                        # mask_array[y1:y2, x1:x2] = template_mask * count_temp_cumsum[i]

                        break
                    else:
                        i = i + 1

                background_mask = 1.0 - template_mask

                for c in range(0, 3):
                    background[y1:y2, x1:x2, c] = (
                            template_mask * template[:, :, c] + background_mask * background[y1:y2, x1:x2, c])

        return background, mask_array

    def get_train_data(self, amount):
        aug_imgs = []
        aug_masks = []

        # data augmentation for background, template, and template mask
        for i in range(amount):
            background_name = self.get_random_background()
            background = cv2.imread(os.path.join(self.root, 'backgrounds', background_name), cv2.IMREAD_UNCHANGED)

            background_size = background.shape[0] * background.shape[1]

            stitch_templates = []
            stitch_template_masks = []
            temp_count = []
            for j in range(len(self.imgs_templates)):
                template = cv2.imread(os.path.join(self.root, 'templates', self.imgs_templates[j]),
                                      cv2.IMREAD_UNCHANGED)
                template_size = template.shape[0] * template.shape[1]

                temp_back_rel = template_size / background_size
                temp_normalization = 1 / (temp_back_rel * self.max_temp_back_rel)
                template = cv2.resize(template, dsize=(0, 0), fx=temp_normalization, fy=temp_normalization)
                template_mask = cv2.resize(self.template_masks[j], dsize=(0, 0), fx=temp_normalization,
                                           fy=temp_normalization)


                rand = np.random.randint(1, self.max_templates + 1)

                if rand > 0:
                    augmented_templates, augmented_template_masks = self.augment_templates(template, template_mask,
                                                                                           rand)
                    stitch_templates.append(augmented_templates)
                    stitch_template_masks.append(augmented_template_masks)
                temp_count.append(rand)
                # print(rand)

            aug_img, aug_mask = self.stitch_templates_to_background(background, stitch_templates, stitch_template_masks,
                                                                    temp_count)

            background_angle = np.random.uniform(-self.max_augm_rot, self.max_augm_rot)
            for k in range(len(aug_mask)):
                aug_mask[k] = rotate(aug_mask[k], background_angle, reshape=False)
            aug_img = rotate(aug_img, background_angle, reshape=False)

            aug_imgs.append(aug_img)
            aug_masks.append(aug_mask)

        return aug_imgs, aug_masks


np.random.seed(1234)
dataset = AugmentedDataset(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images'))
imgs, masks = dataset.get_train_data(1)

cv2.imshow('image', imgs[0])
cv2.waitKey()

mask = cv2.threshold(masks[0][0], 0, 255, cv2.THRESH_BINARY)

cv2.imshow('mask', mask[1])
cv2.waitKey()


mask = cv2.threshold(masks[0][1], 0, 255, cv2.THRESH_BINARY)

cv2.imshow('mask', mask[1])
cv2.waitKey()

### MD: Some manual tests, I will delete them later

# root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images')
# print(root)
#
# imgs_templates = list(sorted([f for f in os.listdir(os.path.join(root, 'templates')) if not f.startswith('.')]))
#
# print(imgs_templates[0])
# template = cv2.imread(os.path.join(root, 'templates', imgs_templates[0]), cv2.IMREAD_UNCHANGED)
#
# template = cv2.resize(template, dsize=(0,0), fx=1/5, fy=1/5)
#
# cv2.imshow('image', template)
# cv2.waitKey()
#
# background = cv2.imread('backgrounds/scene3.jpg', cv2.IMREAD_UNCHANGED)
# template = cv2.imread('templates/1.png', cv2.IMREAD_UNCHANGED)
#
# alpha_channel = template[:, :, 3]
# _, template_mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
# template_mask = template_mask.reshape((alpha_channel.shape[0],alpha_channel.shape[1]))

#
# template = cv2.imread('templates/1.png', cv2.IMREAD_COLOR)
#
# back_height = np.shape(background)[0]
# back_width = np.shape(background)[1]
# temp_height = np.shape(template)[0]
# temp_width = np.shape(template)[1]
# y_coord = np.random.randint(0, back_height - temp_height)
# x_coord = np.random.randint(0, back_width - temp_width)
# y1, y2 = y_coord, y_coord + temp_height
# x1, x2 = x_coord, x_coord + temp_width
#
# background_mask = 1.0 - template_mask
#
# for c in range(0, 3):
#     # background[y1:y2, x1:x2, c] = (template_mask * template[:, :, c] + background_mask * background[y1:y2, x1:x2, c])
#     background[y1:y2, x1:x2, c] = (template_mask * template[:, :, c] + background_mask * background[y1:y2, x1:x2, c])
#
# imgS = cv2.resize(background, (960, 540))
#
# cv2.imshow('image', imgS)
# cv2.waitKey()


# img2 = cv2.imread('templates/chocolate.png', cv2.IMREAD_UNCHANGED)
# alpha_channel = img[:,:,3]
# _ , mask_img = cv2.threshold(alpha_channel,254,255,cv2.THRESH_BINARY)
#
# cv2.imshow('image', img2)
# cv2.waitKey()
# #
# # img = rotate(img, 30, reshape=False)
#
# cv2.imshow('image', mask_img)
# cv2.waitKey()
#
# mask_img = cv2.resize(mask_img, dsize=(0,0), fx=0.5, fy=0.5)
#
# cv2.imshow('image', mask_img)
# cv2.waitKey()
#
# mask_img = rotate(mask_img, -45)
#
# cv2.imshow('image', mask_img)
# cv2.waitKey()

# img2 = cv2.imread("template_mask/FudanPed00008_mask.png", cv2.IMREAD_UNCHANGED)
#
# _ , mask_img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY)
#
# cv2.imshow('image', mask_img2)
# cv2.waitKey()
