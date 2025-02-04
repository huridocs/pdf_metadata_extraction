from time import sleep

from rsmq import RedisSMQ

REDIS_HOST = "127.0.0.1"
REDIS_PORT = "6379"


def wait_for_queues():
    queue = RedisSMQ(
        host=REDIS_HOST,
        port=REDIS_PORT,
        qname="information_extraction_tasks",
        quiet=False,
    )

    for i in range(60):
        try:
            queue.getQueueAttributes().exec_command()
            print("Queue is ready")
            return
        except:
            print("Waiting for queue")
            sleep(5)


if __name__ == "__main__":
    wait_for_queues()
