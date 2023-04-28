# Phân Loại Các Danh Lam Thắng Cảnh Tại Việt Nam

## Thành Viên Thực Hiện
- Phạm Minh Thạch (22C15018)  
- Nguyễn Thanh Tùng (22C15023)

## PHÂN LOẠI BẰNG SUPPORT VECTOR MACHINE
### Giới thiệu
Support Vector Machine (SVM) là một thuật toán học máy có giám sát được sử dụng rất phổ biến ngày nay trong các bài toán phân lớp. Ý tưởng của SVM là tìm ra các đường kẻ để phân tách các điểm dữ liệu. Mỗi đường kẽ này sẽ chia không gian thành các miền khác nhau và mỗi miền sẽ chứa một loại dữ liệu.
### Tiền xử lý dữ liệu
Ở bước tiền xử lý, ta kết hợp phương pháp Histogram of Object Gradient và Standard Scale. Trong đó, Histogram of Object Gradient là phương pháp giúp ta lấy ra các đường nét chính của bức ảnh (Hình 1). Standard Scale được sử dụng để chuẩn hoá dữ liệu, sau khi chuẩn hoá giá trị trung bình của dữ liệu là 0. Quá trình này giúp ta làm giảm được kích thước của dữ liệu đầu vào, đồng thời đưa dữ liệu về khoản thuận lợi cho việc tính toán của các thuật toán.
Input gồm 1500 ảnh, thời gian để load ảnh vào dataset là 2m33s.
### Huấn luyện mô hình
Trên tập huấn luyện gồm 1500 ảnh và 30 nhãn, thời gian huấn luyện là 5s.
### Đánh giá mô hình
Trên tập kiểm tra gồm 1500 ảnh và 30 nhãn, thời gian để đánh giá là 4s, độ chính xác accuracy score là 59.47%.
![image](https://user-images.githubusercontent.com/72611701/235125963-fc7ca86e-de70-4921-9a1b-2667ae9d77ab.png)

## PHÂN LOẠI BẰNG DEEP LEARNING
### Giới thiệu
Các mạng Deep Neural Network bao gồm nhiều lớp neuron khác nhau, có khả năng thực hiện các tính toán có độ phức tạp rất cao. Deep Learning hiện đang phát triển rất nhanh và được xem là một trong những bước đột phá lớn nhất trong Machine Learning. Trong phần này, chúng ta sẽ xây dựng hai network khác nhau để thực hiện công việc phân loại danh lam thắng cảnh.

### Xây dựng mô hình thứ nhất
#### Ý tưởng và thư viện
Mô hình đầu tiên được cài đặt với thư viện Pytorch.
#### Tiền xử lý
Ở bước tiền xử lý, ta sẽ chuẩn hoá dữ liệu theo giá trị trung bình và độ lệch chuẩn, để đưa dữ liệu về phân bố xung quanh điểm trung bình là 0. Việc này sẽ giúp ích cho việc tính toán của các thuật toán. Ảnh cũng được scale về kích thước 128x128. Thời gian thực hiện thao tác tiền xử lý là 46s.
#### Huấn luyện mô hình
Đầu vào của mô hình là một ảnh 3x128x128 (channel x width x height). Mô hình gồm 6 lớp Convolution và 2 lớp Linear. Các lớp Convolution được kết hợp với Max Pool, ReLu và Batch Norm để rút trích đặc trưng. Kết quả của lớp Linear được kết hợp với hàm Log Soft Max. Bảng sau cho ta tóm tắt về kiến trúc của model.
```
----------------------------------------------------------------
Layer (type)               	Output Shape         		
=====================================
Conv2d-1         		[-1, 16, 128, 128]             	
ReLU-2         		[-1, 16, 128, 128]               	
BatchNorm2d-3         	[-1, 16, 128, 128]              	
MaxPool2d-4           	[-1, 16, 64, 64]               		
Conv2d-5           	[-1, 32, 64, 64]           		
ReLU-6           		[-1, 32, 64, 64]               		
BatchNorm2d-7           	[-1, 32, 64, 64]              		
MaxPool2d-8           	[-1, 32, 32, 32]              		 
Conv2d-9          	 	[-1, 64, 32, 32]          		
ReLU-10          	 	[-1, 64, 32, 32]               		
BatchNorm2d-11           [-1, 64, 32, 32]             		
MaxPool2d-12           	[-1, 64, 16, 16]               		
Conv2d-13          	[-1, 128, 16, 16]          		
ReLU-14          		[-1, 128, 16, 16]               	
BatchNorm2d-15          	[-1, 128, 16, 16]             		
MaxPool2d-16            	[-1, 128, 8, 8]               		
Conv2d-17            	[-1, 256, 8, 8]         		
ReLU-18            	[-1, 256, 8, 8]               		
BatchNorm2d-19           [-1, 256, 8, 8]             		
MaxPool2d-20            	[-1, 256, 4, 4]               		
Flatten-21                 	[-1, 4096]               		
Linear-22                 	[-1, 1024]       			
```
Mô hình được huấn luyện trên 20 epoch, với hàm loss được sử dụng là Negative Like và phương pháp được sử dụng để tối ưu bộ tham số là Adam Stochastic Optimization với learning rate là 0.001.
Thời gian thực hiện việc training là 18 phút 20 giây.
#### Đánh giá mô hình
Việc đánh giá mô hình được thực hiện trên tập gồm 1500 ảnh, tỉ lệ dự đoán chính xác là 44%, thời gian thực hiện là 41.5 giây.

### Xây dựng mô hình thứ hai
#### Ý tưởng và thư viện
Sử dụng Transfer Learning, với pre-trained model là Xception.
Sử dụng thư viện tensorflow
#### Tiền xử lý
Chia 80% tập train dùng để huấn luyện, 20% cho validation
Dùng các kỹ thuật image augmentations để tăng số lượng dữ liệu cho tập huấn luyện. Ví dụ xoay ảnh ngang, xoay ảnh dọc, phóng to, ...
#### Huấn luyện mô hình
Hình nhận ảnh với 3x224x224 (channel x width x height). Thêm 4 lớp mới lần lượt là 1 Global Average Pooling, 1 lớp dense, 1 lớp dropout và lớp đầu ra.
Mô hình được huấn luyện với 25 epoch, với hàm loss là cross entropy, dùng Adam để tối ưu tham số.
Thời gian huấn luyện là 25 phút cho 25 epoch, mỗi epoch tầm 1 phút trên Macbook M1.
#### Đánh giá mô hình
Độ chính xác trên tập test 1500 ảnh là 87.4%, thời gian thực hiện 1 phút 25 giây.
Bởi vì mô hình có kích thước vượt quá giới hạn cho phép của github, nên ta lưu trữ tại link bên dưới: drive.google.com/file/d/1MLeUKrWHRUchsqIM85KMw2bcIUUg5w6T/view?usp=share_link

## SO SÁNH KẾT QUẢ CỦA SVM VÀ DEEP LEARNING
Mô hình SVM có thời gian huấn luyện và thời gian đánh giá nhanh hơn mô hình Deep Learning. Tuy nhiên, mô hình Deep Learning thứ hai lại cho kết quả dự đoán có độ chính xác cao hơn so với mô hình SVM. Bảng sau đây thể hiện sự khác biệt giữa SVM và hai mô hình Deep Learning.
	Mô hình SVM	Mô hình Deep Learning 1	Mô hình Deep Learning 2 (Transfer Learning)
Thời gian huấn luyện	5 giây	18 phút 20 giây	25 phút
Thời gian đánh giá	4 giây	41.5 giây	1 phút 25 giây
Tỉ lệ dự đoán chính xác	59.47%	44%	87.4%

Với mô hình Deep Learning sử dụng Transfer Learning đạt được kết quả cao là do nó tận dụng được kiến thức được tìm ra bởi mô hình pre-trained (được huấn luyện trên một tập dữ liệu rất lớn, rất nhiều layer)
![image](https://user-images.githubusercontent.com/72611701/235126827-c42803de-3bb6-4f52-81b8-67429c48bf5d.png)
