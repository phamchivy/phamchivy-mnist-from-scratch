#ifndef SOCKET_UTILS_H
#define SOCKET_UTILS_H

int setup_server(int port);
int accept_client(int server_fd);
int connect_to_server(const char* ip, int port);
int send_all(int sockfd, const void* data, int size);
int recv_all(int sockfd, void* buffer, int size);
void socket_close(int sockfd);

#endif
