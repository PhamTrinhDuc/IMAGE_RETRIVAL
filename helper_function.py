
from lib import *
from model import get_model




class Helper:

    def __init__(self):
        self.dataset_dict, self.image_filenames = self.read_image()

    def read_image(self): # => return dict{path: (class, id)}, list[path image]
        
        data_dir = 'database_image'
        dataset_dict = {}
        image_filenames = []

        for idx, folder_name in enumerate(sorted(os.listdir(data_dir))):
            path_folder = os.path.join(data_dir, folder_name)

            for name_image in os.listdir(path_folder):
                path_image = os.path.join(path_folder, name_image)
                image_filenames.append(path_image)
                dataset_dict[path_image] = (folder_name, idx)
        return dataset_dict, image_filenames

    def plot_images(self): # => plot some images in database

        n_rows = 3
        n_cols = 3

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        vis_idx = 0

        for idx in range(n_cols):
            for jdx in range(n_rows):
                ax[idx, jdx].imshow(self.src_images[vis_idx])
                # print(self.src_images[vis_idx].shape)
                ax[idx, jdx].axis('off')

                vis_idx += 1
        plt.show()

    
    def plot_results(self, image_paths: list, grid_size=(3, 3)): # => plot grid images
        num_images = len(image_paths)
        rows, cols = grid_size

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()

        for i, (ax, img_path) in enumerate(zip(axes, image_paths)):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(str(self.dataset_dict[img_path]))
            ax.axis('off')
        
        # If the number of images is less than the number of cells in the grid, hide the excess axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # display grid image
        plt.tight_layout()
        plt.show()

    

    # def create_image_grid(image_paths, grid_size=(3, 3)):
    #     """
    #     Tạo một grid ảnh từ danh sách các đường dẫn ảnh.

    #     Parameters:
    #     - image_paths (list): Danh sách các đường dẫn ảnh.
    #     - grid_size (tuple): Kích thước grid (số hàng, số cột).

    #     Returns:
    #     - Image: Ảnh tổng hợp chứa grid ảnh.
    #     """
    #     # Số lượng ảnh
    #     num_images = len(image_paths)
    #     rows, cols = grid_size

    #     # Mở tất cả các ảnh và lấy kích thước lớn nhất
    #     images = [Image.open(img_path) for img_path in image_paths]
    #     max_width = max(img.width for img in images)
    #     max_height = max(img.height for img in images)

    #     # Tạo ảnh nền trắng cho grid với kích thước lớn nhất
    #     grid_img = Image.new('RGB', (cols * max_width, rows * max_height), (255, 255, 255))

    #     for idx, img in enumerate(images):
    #         if idx >= rows * cols:
    #             break  # Dừng lại nếu đã đạt số lượng ô trong grid

    #         # Tính toán vị trí của ảnh trong grid
    #         row = idx // cols
    #         col = idx % cols
    #         x = col * max_width
    #         y = row * max_height

    #         # Dán ảnh vào vị trí tương ứng trong grid
    #         grid_img.paste(img, (x, y))

    #     return grid_img

