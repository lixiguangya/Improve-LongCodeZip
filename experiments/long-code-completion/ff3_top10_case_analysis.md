# code_compressor_ff3 top10 case analysis

统计源文件：`top10_cases.txt`。80 个排名条目去重后共有 45 个 case。结论围绕“压缩后上下文保留/删除的信息如何影响 ES/EM”，并直接对应 `code_compressor_ff3.py` 的修改。

## 总体判断

第二组 `code_compressor_ff.py` 效果较好，是因为它的 entropy chunk 边界更容易保住局部 completion 形状：函数签名、相邻 sibling 方法、成对赋值、调用模板、字符串/SQL literal、docstring/comment 线索。

第三组 `code_compressor_ff1.py` 换成 `split_code1.py` 后，局部逻辑经常被切碎或被重排为孤立语义块。它在骨架型 case 上有优势，但在需要“相邻模板”和“自然代码顺序”的 case 上明显掉 ES/EM。

`ff3` 的策略是：保留第二组的 entropy chunk 主路径，只把 split_code1 的 dependency/used_by 信息作为辅助特征；对语义块先做质量控制，异常时回退到 entropy；再用 query-tail、receiver/self、literal、assignment/call、结构位置共同打分。

## 第一组 - 第二组

- case397: 答案是 docstring data 字段 `name - aname/ename/gname`。第一组保留大量重复 Response 类字段模板，第二组删掉了这类看似低价值的 docstring 重复，导致 EM=0。说明字段型 docstring 不能一概降权。
- case268: 答案是 `if self.rate == False:`，依赖前面 `flg_resample/desc` 的同类条件模板。第一组保留完整条件段，第二组压缩后局部条件链断裂。
- case19: `StdTimeSynch.__init__` 需要 `super(...)`。第一组保留类初始化模板，第二组没有保住足够相邻构造函数模式。
- case185: 答案 `ctx = dict(self._context)` 依赖 Odoo issue/action 上下文。第一组保留 `_context` 模式，第二组更多保留业务字段，导致初始化上下文缺失。
- case231: 需要 `elif next == 3:`。第一组保留递增分支序列，第二组遗漏 next 分支 skeleton。
- case104: 需要 `if disk_type == "file":`。第一组保留磁盘类型判断区域，第二组压缩后异常处理和 disk_type 分支衔接不足。
- case173: 需要 `mock_request = MagicMock()`。第一组保留测试 helper/mock 上下文，第二组删掉部分测试搭建模板。
- case5: 需要 `recv(1024)`。第一组保留 TCP receive 测试模式，第二组保留的网络上下文不够贴近目标 API。
- case6: 需要 `--out-fifo`。第一组保留 in/out/err fifo 连续常量，第二组缺少 exact sibling，EM 掉到 0。
- case73: 需要 company list comprehension。第一组保留 company/party 相关分支，第二组保留了更宽的业务逻辑但局部变量链弱。
- case44: 需要 axis_fill 中 `'Y'` lambda。第一组保留 X/Y/Z 成对映射，第二组保留不完整，ES 高但 EM=0。
- case303: 需要 bootstrap descriptor event。第一组保留 Tor event 字符串常量群，第二组保留多但目标常量邻接不足。
- case378: 需要 `date2 = ...`。第一组保留 date1/date2 成对赋值，第二组削弱了连续赋值模板。

## 第二组 - 第一组

- case410: 需要 `removeCommentFromLine`。第二组保留 parser while-loop 和 comment stripping 调用，第一组压得更狠，丢掉关键调用链。
- case166: 需要 Odoo search domain 的 `employee_id` 条件。第二组保留 search/domain 连续参数，第一组丢掉 domain 尾部。
- case339: 需要 `cur = self["list"].getCurrent()`。第二组保留同类 UI sibling 方法，第一组只剩弱相关 UI 代码。
- case372: 需要创建 contributor metadata element。第二组保留 OPF metadata setter 的 API 链，第一组缺少 create/set 过渡。
- case82: 需要 `cfg = VersioneerConfig()`。第二组保留 Versioneer 文档、config 类、get_config 框架，第一组破坏函数起始语境。
- case423: 需要 `refresh_token = json_resp[...]`。第二组保留 OAuth json response 字段断言序列，第一组丢目标字段邻域。
- case156: 需要 `for i in range(prev_end, cluster.start):`。第二组保留 cluster gap 处理，第一组压缩后循环上下文不足。
- case264: 需要遍历 `self["config"].list`。第二组保留 Enigma2 config list UI 模式，第一组保留错区域。
- case429: 需要 pickle protocol loop。第二组保留 pickle test 横向模板，第一组没保住 protocol 循环。
- case75: 需要重复 field 参数 `readonly=True, states=...`。第二组保留 ORM field definition 模板，第一组删掉 exact 参数重复。
- case51: 需要 ansible argument spec 的 `gw4` 字段。第二组保留相邻参数定义，第一组字段模板缺失。
- case80: 需要 `isinstance(prop, basestring)`。第二组保留 getter 类型检查，第一组缺少 prop 验证逻辑。
- case116: 需要 `return S_OK((token, numUses))`。第二组保留 SQL insert 和 token 结果链，第一组断开返回值。

## 第二组 - 第三组：第三组失败重点

- case494: 查询在 `version_lookup_bulk`，答案是 SQL 拼接 `q = ("select %s "`。第二组保留 SQL 构造和函数 framing；第三组虽然有部分 SQL 线索，但语义块压缩后 bulk lookup 的局部顺序/函数边界弱，EM=0。
- case191: 查询尾部有 `# Duh OBT is the reverse`，答案是 `self.target_sz = np.asarray(init_rect[2:])`。第三组删掉注释和 init_rect 相邻状态，导致目标赋值不可恢复。
- case82: get_config 答案 `cfg = VersioneerConfig()`。第三组也可能保留目标类名，但文档/函数入口/配置字段连续性弱，说明 exact token 存在不等于 completion 稳定。
- case165: `get_makeargs` 需要字符串模板拼接。第三组切掉 buildscript/self.makeargs 的 sibling 模式，保留骨架不足以恢复精确字符串。
- case266: `showRightClickMenu` 需要 `if nodeObject.nodeType() == 'PROJECT':`。第三组 ratio 从 3.25 提到 4.07，但删除 selection/index/nodeObject 的相邻控制结构，ES/EM 大降。
- case480: `world_forces_setup` 需要 `func = local_func[1]`。第三组保留 `local_func[0]` 但缺少 else 邻域和 sibling setup 规律，无法推出 `[1]`。
- case339: `timerAdd` 需要 `self["list"].getCurrent()`。第三组没有保留第二组中的 exact sibling，反而留下更远的 `getCurrentSelection`，这是语义块切碎局部 UI 模板的典型失败。
- case263: `testPMS` 需要 password wildcard 检查。第三组保留 notifier/Plex 若干线索，但密码默认值和 `set('*')` 检查邻域被压掉。
- case482: `test_write_bytearray` 需要 `bytearray(b'data')`。第三组将 asyncio write 测试压成 stub，删掉 bytearray 示例。
- case211: `oper` 需要 IRC `send_raw("OPER %s %s"...`。第三组删掉 command 方法群和 docstring，无法迁移 sibling command 模板。
- case45: `set_mesh_grid` 需要 `self.Y = numpy.mgrid[...]`。第三组 ES 高但 EM=0，说明大意还在，精确成对赋值 `self.X/self.Y` 没保住。
- case160: `execute_blind` 需要 `call_name = action.get('call', 'inject')`。第三组保留 action 相关字段但缺少后续调用模板，字段级精确补全失败。
- case174: `forward_patches` 需要 `if head_tree == bottom_tree:`。第三组破坏 `head_tree/bottom_tree` 连续判断，控制流局部连续性不足。

## 第三组 - 第二组：第三组成功点

- case225: POSIX dir_fd 测试需要 `posix.open(posix.getcwd(), ...)`。第三组保留紧凑 API 骨架和相邻测试流程，去掉噪声后 ES/EM=100。
- case397: docstring data 字段补全。第三组保留重复 Response 类字段模板，第二组误删。
- case198: `_send_volume_changed` 需要 `uuid.uuid4().hex`。第三组保留消息发送骨架和 uuid 模式。
- case231: `runTest` 需要递增分支 `elif next == 3:`。第三组保留分支 skeleton，第二组丢掉。
- case446: 需要连续字段 `.strip()`。第三组保留字段清洗模板，第二组上下文更宽但邻接弱。
- case19: `__init__` super 调用。第三组保留初始化骨架，第二组缺 sibling 初始化模板。
- case157: Odoo wizard 需要 `CompanyObj.search([])`。第三组保留 ORM 对象和 search 模式。
- case6: `_drt_cmd_line` 需要 `--out-fifo`。第三组保留 fifo 列表模式，第二组缺 exact sibling。
- case204: PDB parser 需要 `if line[j][:4] == 'pdb|':`。第三组保留循环和前缀判断局部结构。
- case194: `get_storer` 需要 `group = self.get_node(key)`。第三组保留 key/node/storer 过渡。
- case372: metadata setter 需要 create contributor。第三组保留 XML metadata 操作链，EM 成功。
- case0/case1: EM 同为 0，但第三组 ratio 更高；这类说明语义块可以用于低风险激进压缩，不能用来指导高风险 completion。

## split_code1 导致第三组整体变差的机制

- 语义块边界过碎：一行赋值、一行调用、一行控制头被拆开后，模型看不到“上一行到下一行”的自然 continuation。
- 低价值块挤占预算：一些独立控制头、低依赖块、孤立注释被保留，但 sibling 模板和成对赋值被挤掉。
- dependency/used_by 没有正确补偿 PPL：依赖高不等于对当前 query completion 有用；`case339/case482/case211` 都需要 sibling 模板而非 PDG 强依赖。
- docstring/comment 被过度降权：`case191/case397/case82` 显示注释和文档往往是答案格式或目标语义的直接提示。
- 局部顺序破坏：`case480/case174/case45` 的答案依赖 else、成对变量、成对赋值，孤立语义块会破坏这种连续性。

## ff3 修改方案

- 使用 entropy chunk 作为唯一输出选择单元，避免 split_code1 块直接破坏局部顺序。
- 对 split_code1 结果做质量报告：覆盖率、碎片率、单行块比例、非零依赖比例；低覆盖或低信号碎片化时不使用语义依赖 boost。
- 将语义 dependency 投影到 entropy chunk，并对碎片化结果做邻域平滑，相当于合并过碎依赖。
- 细粒度 score 结合 AMI/PPL、dependency、query overlap、query-tail overlap、receiver/self match、结构位置、literal、assignment/call。
- 低风险块更激进压缩：默认 `fine_budget_tighten=0.76`，`context_budget=+0`。
- 高风险块质量保护：query/docstring/literal/assignment/call/receiver 高风险时提高有效保留比例。
- knapsack 后做小预算 repair：在 `1.04 * target_func_tokens` 内补回 query-tail、self/receiver、literal、assignment/call 关键块。
- comment-only 块默认不无条件保留，只有与 query 有重叠或用户显式开启时保留。
