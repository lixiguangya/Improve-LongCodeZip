# code_compressor_ff2 case analysis

本文按 top10_cases.txt 的重复 case 去重分析。`ratio` 是原始 token / 压缩 token，越大表示压缩越狠。

## 总结

第三组 `code_compressor_ff1.py` 的核心问题不是语义块本身无效，而是把 `split_code1.py` 产生的语义块当作主要细粒度压缩单元后，局部 completion 线索被切得过碎：相邻 sibling 方法、docstring/comment、赋值/调用模板、字符串常量和控制流前后文更容易被删。第二组 `code_compressor_ff.py` 的 entropy chunk 在这些 case 中更像“局部语境保护层”，所以即使压缩率低一些，ES/EM 更稳。

`code_compressor_ff2.py` 的目标是保留第二组的 entropy chunk 主路径，把 `split_code1.py` 只作为依赖特征投影到 entropy chunk 上；同时对低风险块压得更紧，对 query 高重合、docstring/comment、赋值/调用/字面量模式块放松预算。

## 第二组优于第三组的 case

- case494: 查询落在 `version_lookup_bulk`，答案是 SQL 拼接起始 `q = ("select %s "`。第二组保留了版本查询函数框架和 SQL 构造；第三组虽然能看到部分 SQL 线索，但语义块压缩后函数 framing 和局部顺序弱，导致模型没有把 bulk lookup 的下一行接出来。
- case191: 查询尾部包含 `# Duh OBT is the reverse`，答案是 `self.target_sz = np.asarray(init_rect[2:])`。第二组保留注释和 `init_rect` 局部状态；第三组把注释/docstring 类线索删得更多，错过“reverse 后继续设置 target_sz”的语义提示。
- case82: `get_config` 的答案是 `cfg = VersioneerConfig()`。第二组完整保留 Versioneer 文档、config 类和函数框架；第三组也有部分目标行，但压缩后文档/函数边界弱，说明 exact line 出现不等于 completion 稳定。
- case165: `get_makeargs` 需要 `makeargs = ' %s %s' % ...`。第二组留下 buildscript/self.makeargs 的 sibling 模式；第三组更多只剩骨架，模板字符串和 self 字段上下文不足。
- case266: `showRightClickMenu` 需要 `if nodeObject.nodeType() == 'PROJECT':`。第二组保留 selection/index/nodeObject 菜单上下文；第三组压缩率从 3.25 提到 4.07，但少了相邻控制结构，EM 掉到 0。
- case480: `world_forces_setup` 需要 `func = local_func[1]`。第二组保留 `local_func, gates, params` 与 force setup 的 sibling 模式；第三组剥掉注释和部分相邻分支，无法从 `else:` 推到 `[1]`。
- case339: `timerAdd` 需要 `cur = self["list"].getCurrent()`。第二组保留 sibling 方法里的完全同形代码；第三组只留下更远的 `getCurrentSelection` 一类弱线索，这是 split_code1 过度切块造成的典型失败。
- case263: `testPMS` 需要 `if set('*') == set(password or '*'):`。第二组保留 Plex notifier/password 检查风格；第三组虽然有类似测试代码，但周围断裂，密码默认值模式不稳。
- case482: `test_write_bytearray` 需要 `data = bytearray(b'data')`。第二组保留 asyncio write tests 中多种 `b'data'` 例子；第三组把测试函数压成 stub，bytearray 示例缺失。
- case211: `oper` 需要 `self.send_raw("OPER %s %s" % (nick, password))`。第二组保留 IRC command 方法群和 docstring；第三组删掉更多 command 模板，无法迁移 `NAMES/JOIN` 这类 sibling 格式。
- case45: `set_mesh_grid` 需要第二个 `numpy.mgrid` 赋给 `self.Y`。第三组 ES 高但 EM 为 0，说明大意保留了；失败点是局部成对赋值 `self.X`/`self.Y` 没被完整保护。
- case160: `execute_blind` 需要 `call_name = action.get('call', 'inject')`。第三组保留了 action 相关块但缺少后续调用模板，模型知道字段却没输出精确赋值。
- case174: `forward_patches` 需要 `if head_tree == bottom_tree:`。第二组保留 patch/tree 状态流；第三组的语义压缩把 commit/tree 相邻判断削弱，控制流模式断了。

## 第三组优于第二组的 case

- case225: `test_mknod_dir_fd` 需要 `f = posix.open(...)`。第三组保留 POSIX dir_fd 测试骨架和相邻 API 调用，压缩率也更高；第二组保留了更多噪声但关键测试流程弱。
- case397: docstring data 字段补全 `name - aname/ename/gname`。第三组保留重复 response 类的结构化字段，第二组没有保住当前数据项模板。
- case198: `_send_volume_changed` 需要 `messageId = uuid.uuid4().hex`。第三组保留消息发送骨架与 uuid 模式，第二组局部 pattern 不足。
- case231: `runTest` 分支需要 `elif next == 3:`。第三组保留紧凑分支 skeleton，第二组压缩后遗漏了递增分支序列。
- case446: 需要 `expected_arrival_date = expected_arrival_date.strip()`。第三组保留连续字段 strip 模板，第二组保留上下文更多但目标字段邻接弱。
- case19: `StdTimeSynch.__init__` 需要 super 调用。第三组保留类继承/初始化骨架，第二组把可迁移初始化模板弱化。
- case157: Odoo wizard 需要 `companies = CompanyObj.search([])`。第三组保留 ORM 对象和 search 模式，第二组散落更多 view 代码。
- case6: `_drt_cmd_line` 需要 `--out-fifo`。第三组保留 in/out/err fifo 常量和列表追加模式，第二组缺少 exact sibling。
- case204: PDB parser 需要 `if line[j][:4] == 'pdb|':`。第三组保留局部循环和字符串前缀判断，第二组保留过宽。
- case194: `get_storer` 需要 `group = self.get_node(key)`。第三组保留 HDF/key/storer 局部语义，第二组没有抓住 key 到 node 的过渡。
- case372: OPF metadata setter 需要创建 contributor 元素。第三组 EM 成功，说明语义块能在重复 XML metadata 操作中保留关键 API 链。
- case0/case1: EM 都为 0，但第三组在相同 ES 下 ratio 更高；这类适合更激进压缩，但不能拿它们指导高风险 completion。

## 第一组和第二组互胜的 case

- 第一组优于第二组: case397, case268, case19, case185, case231, case104, case173, case5, case6, case73, case44, case303, case378。共同点是原始压缩更常保留长重复模板、license/docstring 或完整 sibling 方法；第二组的细粒度策略有时把“看起来低信息量”的重复文本删掉，但这些文本正是答案模板。
- 第二组优于第一组: case410, case166, case339, case372, case82, case423, case156, case264, case429, case75, case51, case80, case116。共同点是第二组对 query 相关函数、相邻结构、条件/赋值片段的排序更好；第一组压得更狠或保留错区域，尤其在 ORM、XML metadata、parser、Versioneer 这类长文件中容易丢目标函数邻域。

## ff2 改进点

- entropy chunk 仍是主细粒度单元，避免 split_code1 直接决定输出边界。
- split_code1 的 dependency count 只投影成打分特征，不直接替代 chunk。
- 新增 query risk，对 query-overlap、docstring/comment、赋值/调用、字符串/SQL 等块提高有效保留比例。
- 默认 coarse context budget 从 `+100` 改为 `+0`，减少粗粒度额外余量。
- 低风险 fine ratio 乘以 `0.82`；中风险至少保留 assigned ratio 的 `0.90`；高风险至少保留到 `0.72` 或 assigned 中较小者。
- comment-only 块默认不再无条件保留，只有与 query overlap 时保留；这回收一部分 token 给真正相关的代码块。
- hybrid score 中提高 query overlap、literal、assignment/call 的权重，并把 semantic dependency 保持为辅助项。
