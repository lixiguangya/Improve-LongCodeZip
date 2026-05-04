def _find_module_utils(module_name, b_module_data, module_path, module_args, task_vars, module_compression, async_timeout, become,
                       become_method, become_user, become_password, environment):

    module_substyle = module_style = 'old'

    if _is_binary(b_module_data):
        module_substyle = module_style = 'binary'

    elif REPLACER in b_module_data:
        # Do REPLACER before from ansible.module_utils because we need make sure
        # we substitute "from ansible.module_utils basic" for REPLACER
        module_style = 'new'
        module_substyle = 'python'
        b_module_data = b_module_data.replace(REPLACER, b'from ansible.module_utils.basic import *')

    elif b'from ansible.module_utils.' in b_module_data:
        module_style = 'new'
        module_substyle = 'python'

    elif REPLACER_WINDOWS in b_module_data:
        module_style = 'new'
        module_substyle = 'powershell'
        b_module_data = b_module_data.replace(REPLACER_WINDOWS, b'#Requires -Module Ansible.ModuleUtils.Legacy')

    elif re.search(b'#Requires \-Module', b_module_data, re.IGNORECASE) \
            or re.search(b'#Requires \-Version', b_module_data, re.IGNORECASE)\
            or re.search(b'#AnsibleRequires \-OSVersion', b_module_data, re.IGNORECASE):
        module_style = 'new'
        module_substyle = 'powershell'

    elif REPLACER_JSONARGS in b_module_data:
        module_style = 'new'
        module_substyle = 'jsonargs'

    elif b'WANT_JSON' in b_module_data:
        module_substyle = module_style = 'non_native_want_json'

    shebang = None

    if module_style in ('old', 'non_native_want_json', 'binary'):
        return b_module_data, module_style, shebang

    output = BytesIO()
    py_module_names = set()

    if module_substyle == 'python':

        params = dict(ANSIBLE_MODULE_ARGS=module_args,)
        python_repred_params = repr(json.dumps(params))

        try:
            compression_method = getattr(zipfile, module_compression)

        except AttributeError:
            display.warning(u'Bad module compression string specified: %s.  Using ZIP_STORED (no compression)' % module_compression)
            compression_method = zipfile.ZIP_STORED

        lookup_path = os.path.join(C.DEFAULT_LOCAL_TMP, 'ansiballz_cache')
        cached_module_filename = os.path.join(lookup_path, "%s-%s" % (module_name, module_compression))
        zipdata = None

        if os.path.exists(cached_module_filename):
            display.debug('ANSIBALLZ: using cached module: %s' % cached_module_filename)
            zipdata = open(cached_module_filename, 'rb').read()

        else:

            if module_name in action_write_locks.action_write_locks:
                display.debug('ANSIBALLZ: Using lock for %s' % module_name)
                lock = action_write_locks.action_write_locks[module_name]

            else:
                # If the action plugin directly invokes the module (instead of
                # going through a strategy) then we don't have a cross-process
                # Lock specifically for this module.  Use the "unexpected
                # module" lock instead
                display.debug('ANSIBALLZ: Using generic lock for %s' % module_name)
                lock = action_write_locks.action_write_locks[None]

            display.debug('ANSIBALLZ: Acquiring lock')

            with lock:

                display.debug('ANSIBALLZ: Lock acquired: %s' % id(lock))

                if not os.path.exists(cached_module_filename):

                    display.debug('ANSIBALLZ: Creating module')

    # ... 

                    zf = zipfile.ZipFile(zipoutput, mode='w', compression=compression_method)

    # ... 

                    zf.writestr('ansible_module_%s.py' % module_name, b_module_data)

                    py_module_cache = {('__init__',): (b'', '[builtin]')}

                    recursive_finder(module_name, b_module_data, py_module_names, py_module_cache, zf)

    # ... 

                    zipdata = base64.b64encode(zipoutput.getvalue())

                    if not os.path.exists(lookup_path):
                        # Note -- if we have a global function to setup, that would
                        # be a better place to run this
                        os.makedirs(lookup_path)

                    display.debug('ANSIBALLZ: Writing module')

                    with open(cached_module_filename + '-part', 'wb') as f:
                        f.write(zipdata)

                    display.debug('ANSIBALLZ: Renaming module')

                    os.rename(cached_module_filename + '-part', cached_module_filename)

                    display.debug('ANSIBALLZ: Done creating module')

            if zipdata is None:

                display.debug('ANSIBALLZ: Reading module after lock')

                try:
                    zipdata = open(cached_module_filename, 'rb').read()

                except IOError:
                    raise AnsibleError('A different worker process failed to create module file. '
                                       'Look at traceback for that process for debugging information.')

        zipdata = to_text(zipdata, errors='surrogate_or_strict')
        shebang, interpreter = _get_shebang(u'/usr/bin/python', task_vars)

        if shebang is None:
            shebang = u'#!/usr/bin/python'

        interpreter_parts = interpreter.split(u' ')
        interpreter = u"'{0}'".format(u"', '".join(interpreter_parts))
        now = datetime.datetime.utcnow()

        output.write(to_bytes(ACTIVE_ANSIBALLZ_TEMPLATE % dict(
            zipdata=zipdata,
            ansible_module=module_name,
            params=python_repred_params,
            shebang=shebang,
            interpreter=interpreter,
            coding=ENCODING_STRING,
            year=now.year,
            month=now.month,
            day=now.day,
            hour=now.hour,
            minute=now.minute,
            second=now.second,
        )))

        b_module_data = output.getvalue()

    elif module_substyle == 'powershell':
        # Powershell/winrm don't actually make use of shebang so we can
        # safely set this here.  If we let the fallback code handle this
        # it can fail in the presence of the UTF8 BOM commonly added by
        # Windows text editors

        shebang = u'#!powershell'

        exec_manifest = dict(
            module_entry=to_text(base64.b64encode(b_module_data)),
            powershell_modules=dict(),
            module_args=module_args,
            actions=['exec'],
            environment=environment
        )

        exec_manifest['exec'] = to_text(base64.b64encode(to_bytes(leaf_exec)))

        if async_timeout > 0:

            exec_manifest["actions"].insert(0, 'async_watchdog')
            exec_manifest["async_watchdog"] = to_text(base64.b64encode(to_bytes(async_watchdog)))
            exec_manifest["actions"].insert(0, 'async_wrapper')
            exec_manifest["async_wrapper"] = to_text(base64.b64encode(to_bytes(async_wrapper)))
            exec_manifest["async_jid"] = str(random.randint(0, 999999999999))
            exec_manifest["async_timeout_sec"] = async_timeout

        if become and become_method == 'runas':
            exec_manifest["actions"].insert(0, 'become')
            exec_manifest["become_user"] = become_user
            exec_manifest["become_password"] = become_password
            exec_manifest["become"] = to_text(base64.b64encode(to_bytes(become_wrapper)))

        lines = b_module_data.split(b'\n')
        module_names = set()
        become_required = False
        min_os_version = None
        min_ps_version = None

        requires_module_list = re.compile(to_bytes(r'(?i)^#\s*requires\s+\-module(?:s?)\s*(Ansible\.ModuleUtils\..+)'))
        requires_ps_version = re.compile(to_bytes('(?i)^#requires\s+\-version\s+([0-9]+(\.[0-9]+){0,3})$'))
        requires_os_version = re.compile(to_bytes('(?i)^#ansiblerequires\s+\-osversion\s+([0-9]+(\.[0-9]+){0,3})$'))
        requires_become = re.compile(to_bytes('(?i)^#ansiblerequires\s+\-become$'))

        for line in lines:

            module_util_line_match = requires_module_list.match(line)

            if module_util_line_match:
                module_names.add(module_util_line_match.group(1))

            requires_ps_version_match = requires_ps_version.match(line)

            if requires_ps_version_match:
                min_ps_version = to_text(requires_ps_version_match.group(1))
                # Powershell cannot cast a string of "1" to version, it must
                # have at least the major.minor for it to work so we append 0
                if requires_ps_version_match.group(2) is None:
                    min_ps_version = "%s.0" % min_ps_version

            requires_os_version_match = requires_os_version.match(line)

            if requires_os_version_match:
                min_os_version = to_text(requires_os_version_match.group(1))
                if requires_os_version_match.group(2) is None:
                    min_os_version = "%s.0" % min_os_version

            requires_become_match = requires_become.match(line)

            if requires_become_match:
                become_required = True

        for m in set(module_names):

            m = to_text(m)
            mu_path = ps_module_utils_loader.find_plugin(m, ".psm1")

            if not mu_path:
                raise AnsibleError('Could not find imported module support code for \'%s\'.' % m)

            exec_manifest["powershell_modules"][m] = to_text(
                base64.b64encode(
                    to_bytes(
                        _slurp(mu_path)
                    )
                )
            )

        exec_manifest['min_ps_version'] = min_ps_version
        exec_manifest['min_os_version'] = min_os_version

        if become_required and 'become' not in exec_manifest["actions"]:
            exec_manifest["actions"].insert(0, 'become')
            exec_manifest["become_user"] = "SYSTEM"
            exec_manifest["become_password"] = None
            exec_manifest["become"] = to_text(base64.b64encode(to_bytes(become_wrapper)))

        module_json = json.dumps(exec_manifest)
        b_module_data = exec_wrapper.replace(b"$json_raw = ''", b"$json_raw = @'\r\n%s\r\n'@" % to_bytes(module_json))

    elif module_substyle == 'jsonargs':

        module_args_json = to_bytes(json.dumps(module_args))

        python_repred_args = to_bytes(repr(module_args_json))
        b_module_data = b_module_data.replace(REPLACER_VERSION, to_bytes(repr(__version__)))
        b_module_data = b_module_data.replace(REPLACER_COMPLEX, python_repred_args)
        b_module_data = b_module_data.replace(REPLACER_SELINUX, to_bytes(','.join(C.DEFAULT_SELINUX_SPECIAL_FS)))

        b_module_data = b_module_data.replace(REPLACER_JSONARGS, module_args_json)
        facility = b'syslog.' + to_bytes(task_vars.get('ansible_syslog_facility', C.DEFAULT_SYSLOG_FACILITY), errors='surrogate_or_strict')
        b_module_data = b_module_data.replace(b'syslog.LOG_USER', facility)

    return (b_module_data, module_style, shebang)