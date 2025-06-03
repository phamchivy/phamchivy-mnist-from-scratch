# Sử dụng Ubuntu nhẹ
FROM ubuntu:22.04

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    make \
    libpthread-stubs0-dev \
    netcat \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Sao chép mã nguồn vào container
WORKDIR /app
COPY . .

# Biên dịch với Makefile
RUN make app

# Đảm bảo file thực thi có tên là 'app'
# Nếu Makefile sinh ra tên khác (ví dụ main), đổi tên
RUN [ -f main ] && mv main app || true

# Mặc định không chạy gì vì docker-compose sẽ cung cấp command
CMD ["./app"]
