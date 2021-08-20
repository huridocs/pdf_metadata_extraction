import asyncio
from information_extraction.InformationExtraction import InformationExtraction


async def calculate_suggestions():
    InformationExtraction.calculate_suggestions_from_database()
    await asyncio.sleep(5)


def loop_calculate_suggestions():
    while True:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(calculate_suggestions())


if __name__ == '__main__':
    loop_calculate_suggestions()
