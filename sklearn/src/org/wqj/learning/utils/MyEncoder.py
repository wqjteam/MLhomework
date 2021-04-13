# -*- coding:utf-8 -*-
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

from src.org.wqj.learning.AITheory.ListToTree import TreeNode


class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        """
        只要检查到了是bytes类型的数据就把它转为str类型
        :param obj:
        :return:
        """
        if isinstance(obj, TreeNode):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)