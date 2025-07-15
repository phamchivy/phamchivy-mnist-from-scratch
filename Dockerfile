# Multi-stage build for Hybrid EASGD System Only
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    libgomp1 \
    libpthread-stubs0-dev \
    build-essential \
    gdb \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY matrix/ ./matrix/
COPY neural/ ./neural/
COPY util/ ./util/
COPY socket/ ./socket/
COPY pipeline/ ./pipeline/
COPY *.c ./
COPY *.h ./
COPY Makefile ./

# Build hybrid system only
RUN make clean && make app_hybrid

# Final runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    iproute2 \
    iputils-ping \
    netcat-openbsd \
    tcpdump \
    iperf3 \
    htop \
    curl \
    procps \
    net-tools \
    bc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy hybrid executable only from builder stage
COPY --from=builder /app/app_hybrid ./

# Copy necessary runtime files
COPY --from=builder /app/matrix/ ./matrix/
COPY --from=builder /app/neural/ ./neural/
COPY --from=builder /app/util/ ./util/
COPY --from=builder /app/socket/ ./socket/
COPY --from=builder /app/pipeline/ ./pipeline/

# Create directories for data and results
RUN mkdir -p /app/data /app/results /app/logs

# Copy shell scripts from external files
COPY start_hybrid.sh /app/
COPY health_check.sh /app/
COPY rotate_logs.sh /app/

# Set appropriate permissions
RUN chown -R appuser:appuser /app && \
    chmod +x /app/app_hybrid && \
    chmod +x /app/start_hybrid.sh /app/health_check.sh /app/rotate_logs.sh

# Install log rotation as cron job (simplified)
RUN echo "0 */6 * * * /app/rotate_logs.sh" > /etc/crontab

# Switch to app user for security
USER appuser

# Set environment variables
ENV NETWORK_TYPE=LOCAL
ENV LOG_LEVEL=1
ENV ALPHA1=0.2
ENV ALPHA2=0.3
ENV BETA=0.5

# Expose ports for the hybrid system
# 8001-8004: Pipeline communication ports
# 12347: Statistics collection port
# 12348: EASGD communication port
EXPOSE 8001 8002 8003 8004 12347 12348

# Set default command
CMD ["./start_hybrid.sh", "--help"]

# Add metadata labels
LABEL version="2.0"
LABEL description="Hybrid Pipeline-EASGD Neural Network Training System"
LABEL maintainer="Research Team"
LABEL architecture="hybrid-easgd"
LABEL network.type="distributed"
LABEL algorithm.easgd="true"