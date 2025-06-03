#include "socket_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <errno.h>

int setup_server(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 1);

    return server_fd;
}

int accept_client(int server_fd) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    return accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
}

int connect_to_server(const char* ip, int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &addr.sin_addr);
    
    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        exit(1);
    }

    return sockfd;
}

int send_all(int sockfd, const void* data, int size) {
    const char* ptr = (const char*)data;
    int total = 0;
    while (total < size) {
        int sent = send(sockfd, ptr + total, size - total, 0);
        if (sent <= 0) return -1;
        total += sent;
    }
    return total;
}

int recv_all(int sockfd, void* buffer, int size) {
    char* ptr = (char*)buffer;
    int total = 0;
    while (total < size) {
        int recvd = recv(sockfd, ptr + total, size - total, 0);
        if (recvd <= 0) return -1;
        total += recvd;
    }
    return total;
}

void socket_close(int sockfd) {
    close(sockfd);
}

int send_matrix(int sockfd, Matrix* mat) {
    // Gửi rows và cols trước
    if (send_all(sockfd, &mat->rows, sizeof(int)) <= 0) return -1;
    if (send_all(sockfd, &mat->cols, sizeof(int)) <= 0) return -1;

    // Gửi entries
    for (int i = 0; i < mat->rows; i++) {
        if (send_all(sockfd, mat->entries[i], sizeof(double) * mat->cols) <= 0) return -1;
    }

    return 0;
}

Matrix* recv_matrix(int sockfd) {
    int rows, cols;
    if (recv_all(sockfd, &rows, sizeof(int)) <= 0) return NULL;
    if (recv_all(sockfd, &cols, sizeof(int)) <= 0) return NULL;

    Matrix* mat = malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->entries = malloc(rows * sizeof(double*));

    for (int i = 0; i < rows; i++) {
        mat->entries[i] = malloc(cols * sizeof(double));
        if (recv_all(sockfd, mat->entries[i], sizeof(double) * cols) <= 0) return NULL;
    }

    return mat;
}

int send_loss(int sockfd, double loss) {
    return send_all(sockfd, &loss, sizeof(double));
}

double recv_loss(int sockfd) {
    double loss;
    if (recv_all(sockfd, &loss, sizeof(double)) <= 0) return -1;
    return loss;
}