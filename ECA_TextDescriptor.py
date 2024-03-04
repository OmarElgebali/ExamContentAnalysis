from openai import OpenAI

client = OpenAI(api_key='')


def describe_row(row):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "user",
                "content": f"Describe briefly the following:\n {row}"
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


def describe_dataset(dataset):
    described_dataset = [describe_row(row) for row in dataset]
    return described_dataset + dataset

