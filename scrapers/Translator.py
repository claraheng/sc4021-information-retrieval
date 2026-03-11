import asyncio
import pandas as pd
from googletrans import Translator
from tqdm import tqdm

FILE_NAME = 'bilibili_comments_1.xlsx'


async def translate_all():
    # 读取文件
    df = pd.read_excel(FILE_NAME)
    print(f"📊 共 {len(df)} 条评论")
    print(f"📋 列名: {list(df.columns)}")

    # 创建翻译器
    translator = Translator()

    # 逐条翻译 - 使用第一列（评论内容）
    for i in tqdm(range(len(df)), desc="翻译进度"):
        chinese = str(df.iloc[i, 0])  # 使用 iloc 按位置取第一列
        if chinese and chinese != 'nan':
            try:
                translated = await translator.translate(chinese, src='zh-cn', dest='en')
                df.loc[i, 'comment_en'] = translated.text
            except Exception as e:
                df.loc[i, 'comment_en'] = f'[翻译失败: {str(e)[:30]}]'
        else:
            df.loc[i, 'comment_en'] = ''

    # 保存
    df.to_excel('bilibili_comments_1_translated.xlsx', index=False)
    print("✅ 完成！")


# 运行
asyncio.run(translate_all())

