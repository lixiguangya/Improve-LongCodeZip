

def _retrieve_schema_from_server(self, url, conn=None):
    tmpdir = None
    has_conn = conn is not None

    self.debug('retrieving schema for SchemaCache url=%s conn=%s', url, conn)

    try:
        if api.env.context == 'server' and conn is None:
            # FIXME: is this really what we want to do?
            # This seems like this logic is in the wrong place and may conflict with other state.
            try:
                # Create a new credentials cache for this Apache process
                tmpdir = tempfile.mkdtemp(prefix = "tmp-")
                ccache_file = 'FILE:%s/ccache' % tmpdir
                krbcontext = krbV.default_context()
                principal = str('HTTP/%s@%s' % (api.env.host, api.env.realm))
                keytab = krbV.Keytab(name='/etc/httpd/conf/ipa.keytab', context=krbcontext)
                principal = krbV.Principal(name=principal, context=krbcontext)
                prev_ccache = os.environ.get('KRB5CCNAME')
                os.environ['KRB5CCNAME'] = ccache_file
                ccache = krbV.CCache(name=ccache_file, context=krbcontext, primary_principal=principal)
                ccache.init(principal)
                ccache.init_creds_keytab(keytab=keytab, principal=principal)
            except krbV.Krb5Error, e:
                raise StandardError('Unable to retrieve LDAP schema. Error initializing principal %s in %s: %s' % (principal.name, '/etc/httpd/conf/ipa.keytab', str(e)))
            finally:
                if prev_ccache is not None:
                    os.environ['KRB5CCNAME'] = prev_ccache


        if conn is None:
            conn = IPASimpleLDAPObject(url)
            if url.startswith('ldapi://'):
                conn.set_option(_ldap.OPT_HOST_NAME, api.env.host)
            conn.sasl_interactive_bind_s(None, SASL_AUTH)

        try:
            schema_entry = conn.search_s('cn=schema', _ldap.SCOPE_BASE,
                attrlist=['attributetypes', 'objectclasses'])[0]
        except _ldap.NO_SUCH_OBJECT:
            # try different location for schema
            # openldap has schema located in cn=subschema
            self.debug('cn=schema not found, fallback to cn=subschema')
            schema_entry = conn.search_s('cn=subschema', _ldap.SCOPE_BASE,
                attrlist=['attributetypes', 'objectclasses'])[0]
        if not has_conn:
            conn.unbind_s()
    except _ldap.SERVER_DOWN:
        raise NetworkError(uri=url,
                           error=u'LDAP Server Down, unable to retrieve LDAP schema')
    except _ldap.LDAPError, e:
        desc = e.args[0]['desc'].strip()
        info = e.args[0].get('info', '').strip()
        raise DatabaseError(desc = u'uri=%s' % url,
                            info = u'Unable to retrieve LDAP schema: %s: %s' % (desc, info))
    except IndexError:
        # no 'cn=schema' entry in LDAP? some servers use 'cn=subschema'
        # TODO: DS uses 'cn=schema', support for other server?
        #       raise a more appropriate exception
        raise
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir)

    return _ldap.schema.SubSchema(schema_entry[1])
