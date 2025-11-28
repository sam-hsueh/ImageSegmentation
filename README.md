这是一个标注、训练、实例分割于一体的方案;

训练和预测部分是按https://github.com/IntptrMax/YoloSharp 的代码添加了UI;

UI框架是https://github.com/MaterialDesignInXAML/MaterialDesignInXamlToolkit;

Download the file from https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.9.0%2Bcu130.zip
After unzipping the above file, copy the DLL files under the folder ...libtorch-win-shared-with-deps-2.9.0+cu130\libtorch\lib to the folder ...\bin...runtimes\win-x64\native;

Download the file from https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth;
Copy it to the SamModels folder;

Reference all files in the DLL directory;

Annotation, training, prediction:

<img width="1920" height="1140" alt="屏幕截图 2025-11-12 120104" src="https://github.com/user-attachments/assets/0649f741-be64-475d-823e-75fc3a9d4728" />

<img width="1920" height="1140" alt="屏幕截图 2025-11-28 234227" src="https://github.com/user-attachments/assets/65c9a070-e149-4a3c-a19b-d60aa50cf300" />

<img width="1920" height="1140" alt="屏幕截图 2025-11-12 120319" src="https://github.com/user-attachments/assets/9dac0810-bfcd-4cb7-9c99-ee7f94df7eb9" />
