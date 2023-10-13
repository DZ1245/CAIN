* **Upload token**: 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
* 原始图片格式为uint16，不符合神经网络需求，处理数据时修改为int32，从内存占用考虑可以换成int16，应该不存在超出取值范围的点。**需要进一步修改为float32，暂时用.to(dtype=torch.float32),后续重做数据集**
* test data loader 中存在一个test_mode，biology中尚未实现并了解作用
    ```python
    # biology的get_loader表现为
    elif dataset_str == 'biology':
        from data.Biology import get_loader
        test_loader = get_loader('test', data_root, test_batch_size, shuffle=False, num_workers=num_workers)

    # 原作者get_loader表现为
    elif dataset_str == 'custom':
        from data.video import get_loader
        test_loader = get_loader('test', data_root, test_batch_size, img_fmt=img_fmt, shuffle=False, num_workers=num_workers, n_frames=1)
        return test_loader
    ```
    需要解决，否则无法进行test
* 