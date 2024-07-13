class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        # tmp = dataset.split("_")
        # if len(tmp) == 2:
        #     dataset = tmp[1]
        # return 'D:\learn_pytorch\data\规则耕地数据集\\' + dataset
        return 'E:\耕地数据集\荷兰\荷兰数据集'


if __name__ == "__main__":
    print(Path.db_root_dir("GID"))