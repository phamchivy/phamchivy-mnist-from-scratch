#include "socket_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/time.h>
#include <errno.h>
#include <fcntl.h>

int setup_server(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return -1;
    }
    
    // Enable address reuse
    int reuse = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("setsockopt");
        close(server_fd);
        return -1;
    }
    
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return -1;
    }
    
    if (listen(server_fd, 1) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }

    return server_fd;
}

int accept_client(int server_fd) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) {
        perror("accept");
        return -1;
    }
    return client_fd;
}

int connect_to_server(const char* ip, int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }
    
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &addr.sin_addr);
    
    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(sockfd);
        return -1;
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

// New functions for pipeline support
int set_socket_nonblocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags == -1) return -1;
    
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) == -1) {
        return -1;
    }
    return 0;
}

int set_socket_blocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags == -1) return -1;
    
    if (fcntl(sockfd, F_SETFL, flags & ~O_NONBLOCK) == -1) {
        return -1;
    }
    return 0;
}

// Check if data is available to read without blocking
int has_pending_data(int sockfd) {
    fd_set readfds;
    struct timeval timeout = {0, 0}; // No wait
    
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);
    
    int result = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
    return result > 0 && FD_ISSET(sockfd, &readfds);
}

// Set socket buffer sizes for better pipeline performance
int set_socket_buffers(int sockfd, int send_buf_size, int recv_buf_size) {
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &send_buf_size, sizeof(send_buf_size)) < 0) {
        return -1;
    }
    
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &recv_buf_size, sizeof(recv_buf_size)) < 0) {
        return -1;
    }
    
    return 0;
}