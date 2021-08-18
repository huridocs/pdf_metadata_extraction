# @app.post('/redis')
# async def redis_post():
#     queue = RedisSMQ(host="redis", qname="segment", )
#     queue.deleteQueue().exceptions(False).execute()
#     queue.createQueue(delay=0).vt(20).execute()
#     queue.sendMessage(delay=0).message('first message').execute()
#     return str(queue.getQueueAttributes().execute())


