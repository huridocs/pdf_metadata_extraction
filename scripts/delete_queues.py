from redis import exceptions
from rsmq import RedisSMQ

REDIS_HOST = "127.0.0.1"
REDIS_PORT = "6379"

queues = ["information_extraction", "extract_paragraphs"]


def delete_queues():
    try:
        for queue_name in queues:
            queue = RedisSMQ(
                host=REDIS_HOST,
                port=REDIS_PORT,
                qname=f"{queue_name}_tasks",
                quiet=False,
            )

            queue.deleteQueue().exceptions(False).execute()
            queue.createQueue().exceptions(False).execute()

            queue = RedisSMQ(
                host=REDIS_HOST,
                port=REDIS_PORT,
                qname=f"{queue_name}_results",
                quiet=False,
            )

            queue.deleteQueue().exceptions(False).execute()
            queue.createQueue().exceptions(False).execute()

            print("Queues properly deleted")

    except exceptions.ConnectionError:
        print("No redis connection")


if __name__ == "__main__":
    delete_queues()
