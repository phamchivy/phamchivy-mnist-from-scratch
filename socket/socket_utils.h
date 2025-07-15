#ifndef SOCKET_UTILS_H
#define SOCKET_UTILS_H

int setup_server(int port);
int accept_client(int server_fd);
int connect_to_server(const char* ip, int port);
int send_all(int sockfd, const void* data, int size);
int recv_all(int sockfd, void* buffer, int size);
void socket_close(int sockfd);

// Pipeline support functions
int set_socket_nonblocking(int sockfd);
int set_socket_blocking(int sockfd);
int has_pending_data(int sockfd);
int set_socket_buffers(int sockfd, int send_buf_size, int recv_buf_size);

#endif
