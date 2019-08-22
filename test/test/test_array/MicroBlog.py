# -*- coding:utf-8 -*-

import time


# 博客类
class Blog:

    def __init__(self, userId, Blogld, postDate):
        self.userId = userId
        self.Blogld = Blogld
        self.postDate = postDate
        self.context = None

    def __str__(self):
        print("userId %s, Blogld %s, postDate %s" % (self.userId, self.Blogld, self.postDate))


# 微博类
class MicroBlog:

    def __init__(self, user_id):
        self.userId = user_id
        # 微博用户发布的微博集合
        self.blogs = []
        # 微博用户关注的用户集合
        self.flows = []

    def PostBlog(self, userld, Blogld):
        self.blogs.append(Blog(userld, Blogld, time.time()))

    def DeleteBlog(self, userld, Blogld):
        for blog in self.blogs:
            # 将微博删除
            if blog.userId == userld and blog.Blogld == Blogld:
                self.blogs.remove(blog)
                break
        # print("此微博用户没有对应的微博")

    def Follow(self, userld, friendld):
        if userld == self.userId and friendld in users.keys():
            self.flows.append(users[friendld])

    def UnFollow(self, userld, friendld):
        if userld == self.userId:
            for flow in self.flows:
                if flow.userId == friendld and friendld in users.keys():
                    self.flows.remove(flow)
                    break
            # print("此微博用户没有关注的当前朋友")

    def GetNewsFeed(self, userld):
        if userld == self.userId:
            result = self.blogs.copy()
            for flow in self.flows:
                result.extend(flow.blogs)
            result.sort(key=lambda x: x.postDate, reverse=True)
            return result[:9] if len(result) > 10 else result


# 所有用户
users = {}
if __name__ == '__main__':
    user1 = MicroBlog(1)
    user2 = MicroBlog(2)
    users[2] = user2
    users[1] = user1

    user1.PostBlog(1, 1)
    time.sleep(1)
    user1.PostBlog(1, 2)

    time.sleep(1)
    user2.PostBlog(2, 1)
    time.sleep(1)
    user2.PostBlog(2, 2)
    user1.Follow(1, 2)
    time.sleep(1)
    user1.PostBlog(1, 3)
    result1 = user1.GetNewsFeed(1)
    for i in result1:
        print(i.userId, i.Blogld, i.postDate)

    print("=" * 20)
    user1.DeleteBlog(1, 1)
    user2.DeleteBlog(2, 1)
    result1 = user1.GetNewsFeed(1)
    for i in result1:
        print(i.userId, i.Blogld, i.postDate)

    print("=" * 20)
    user1.UnFollow(1, 2)
    result1 = user1.GetNewsFeed(1)
    for i in result1:
        print(i.userId, i.Blogld, i.postDate)

    print("=" * 20)
    user2.Follow(2, 1)
    result1 = user2.GetNewsFeed(2)
    for i in result1:
        print(i.userId, i.Blogld, i.postDate)
