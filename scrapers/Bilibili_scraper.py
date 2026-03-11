"""
# link used: https://www.bilibili.com/video/BV1s6ZHBmEpJ/?spm_id_from=333.337.search-card.all.click&vd_source=8e9098d3a47ab3d1db7ea0353c497f04
"""


from DrissionPage import ChromiumPage
import time
import pandas as pd

# 初始化浏览器
browser = ChromiumPage()

# 监听B站评论数据包
browser.listen.start('main?oid=')

# 访问B站视频页面
url = input("url：")
browser.get(url)
time.sleep(3)

# 存储评论数据
comments_list = []
target_comments = 2000  # 目标评论数
no_new_comments_count = 0  # 连续没有新评论的次数
max_no_new_comments = 5  # 最大连续无新评论次数（超过则停止）

print(f"开始抓取评论，目标：{target_comments}条")
print("正在加载评论...")

# 循环直到达到目标评论数或无法加载更多
while len(comments_list) < target_comments:
    # 滚动加载更多评论
    for i in range(10):  # 每次滚动10次
        browser.scroll.down(1000)
        time.sleep(0.3)

    # 等待数据包
    time.sleep(1)

    # 解析当前批次的数据包
    new_comments_count = 0
    for packet in browser.listen.steps(timeout=3):
        try:
            # 检查是否有评论数据
            if 'data' in packet.response.body and 'replies' in packet.response.body['data']:
                for comment in packet.response.body['data']['replies']:
                    # 检查评论是否已存在（简单去重）
                    comment_id = comment.get('rpid', '')
                    if comment_id not in [c[5] for c in comments_list if len(c) > 5]:
                        # 安全地获取各个字段，如果不存在就设为空字符串
                        comment_data = [
                            comment.get('content', {}).get('message', ''),  # 评论内容
                            comment.get('member', {}).get('uname', ''),  # 用户名
                            comment.get('reply_control', {}).get('location', ''),  # IP属地
                            comment.get('like', 0),  # 点赞数
                            comment.get('ctime', ''),  # 评论时间戳
                            comment.get('rpid', '')  # 评论ID（用于去重）
                        ]
                        comments_list.append(comment_data)
                        new_comments_count += 1

                        # 每收集100条打印一次进度
                        if len(comments_list) % 100 == 0:
                            print(f"已收集 {len(comments_list)}/{target_comments} 条评论")

                        # 达到目标后提前退出
                        if len(comments_list) >= target_comments:
                            break

                if len(comments_list) >= target_comments:
                    break

        except Exception as e:
            continue

    # 检查是否还有新评论
    if new_comments_count == 0:
        no_new_comments_count += 1
        print(f"第 {no_new_comments_count} 次未检测到新评论")
    else:
        no_new_comments_count = 0  # 重置计数

    # 如果连续多次没有新评论，说明已经到底了
    if no_new_comments_count >= max_no_new_comments:
        print("已加载完所有评论，无法获取更多")
        break

    # 打印当前进度
    print(f"当前进度：{len(comments_list)}/{target_comments} 条评论，继续加载...")

    # 每次滚动后稍作等待
    time.sleep(1)

# 移除评论ID列（不需要保存到Excel）
comments_for_save = [comment[:-1] for comment in comments_list]

# 保存到Excel
if comments_for_save:
    df = pd.DataFrame(comments_for_save,
                      columns=['评论内容', '用户名', 'IP属地', '点赞数', '时间戳'])

    # 生成文件名（包含时间戳和评论数）
    filename = f'bilibili_comments_{len(comments_for_save)}_{int(time.time())}.xlsx'
    df.to_excel(filename, index=False)

    print(f"\n✅ 完成！共收集 {len(comments_for_save)} 条评论")
    print(f"📁 文件已保存为：{filename}")

    # 显示统计信息
    print(f"\n统计信息：")
    print(f"- 目标评论数：{target_comments}条")
    print(f"- 实际获取：{len(comments_for_save)}条")
    print(f"- 获取比例：{len(comments_for_save) / target_comments * 100:.1f}%")

    # 显示前几条评论作为预览
    print("\n预览前5条评论：")
    for i, comment in enumerate(df['评论内容'].head()):
        print(f"{i + 1}. {comment[:50]}..." if len(comment) > 50 else f"{i + 1}. {comment}")
else:
    print("❌ 没有收集到任何评论，请检查：")
    print("1. 视频链接是否正确")
    print("2. 视频评论区是否开启")
    print("3. 是否需要登录B站")

browser.close()
