FROM registry.agibot.com/genie-sim/openpi_server:latest
COPY . .

# CMD ["/bin/bash", "./scripts/server.sh", "0", "8999"]
