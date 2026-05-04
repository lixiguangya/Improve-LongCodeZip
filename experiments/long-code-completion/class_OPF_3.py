    def __set__(self, obj, val):
        MetadataField.__set__(self, obj, val)
        matches = obj.title_path(obj.metadata)
        if matches:
            for match in matches:
                for attr in list(match.attrib):
                    if attr.endswith('file-as'):
                        del match.attrib[attr]

# ... 



 
class OPF(object): # {{{

    # ... 

    META             = '{%s}meta' % NAMESPACES['opf']
    xpn = NAMESPACES.copy()

    xpn.pop(None)
    xpn['re'] = 'http://exslt.org/regular-expressions'
    XPath = functools.partial(etree.XPath, namespaces=xpn)
    CONTENT          = XPath('self::*[re:match(name(), "meta$", "i")]/@content')
    TEXT             = XPath('string()')
 


    metadata_path   = XPath('descendant::*[re:match(name(), "metadata", "i")]')
    metadata_elem_path = XPath('descendant::*[re:match(name(), concat($name, "$"), "i") or (re:match(name(), "meta$", "i") and re:match(@name, concat("^calibre:", $name, "$"), "i"))]')
    title_path      = XPath('descendant::*[re:match(name(), "title", "i")]')
    authors_path    = XPath('descendant::*[re:match(name(), "creator", "i") and (@role="aut" or @opf:role="aut" or (not(@role) and not(@opf:role)))]')
    bkp_path        = XPath('descendant::*[re:match(name(), "contributor", "i") and (@role="bkp" or @opf:role="bkp")]')
    tags_path       = XPath('descendant::*[re:match(name(), "subject", "i")]')
    isbn_path       = XPath('descendant::*[re:match(name(), "identifier", "i") and '+
                            '(re:match(@scheme, "isbn", "i") or re:match(@opf:scheme, "isbn", "i"))]')
    pubdate_path    = XPath('descendant::*[re:match(name(), "date", "i")]')
    raster_cover_path = XPath('descendant::*[re:match(name(), "meta", "i") and ' +
            're:match(@name, "cover", "i") and @content]')
    identifier_path = XPath('descendant::*[re:match(name(), "identifier", "i")]')
    application_id_path = XPath('descendant::*[re:match(name(), "identifier", "i") and '+
                            '(re:match(@opf:scheme, "calibre|libprs500", "i") or re:match(@scheme, "calibre|libprs500", "i"))]')
    uuid_id_path    = XPath('descendant::*[re:match(name(), "identifier", "i") and '+
                            '(re:match(@opf:scheme, "uuid", "i") or re:match(@scheme, "uuid", "i"))]')
    languages_path  = XPath('descendant::*[local-name()="language"]')
 

    # ... 

    title           = MetadataField('title', formatter=lambda x: re.sub(r'\s+', ' ', x))
    publisher       = MetadataField('publisher')
    comments        = MetadataField('description')
    category        = MetadataField('type')
    rights          = MetadataField('rights')
    series          = MetadataField('series', is_dc=False)
    if tweaks['use_series_auto_increment_tweak_when_importing']:

        series_index    = MetadataField('series_index', is_dc=False,
                                        formatter=float, none_is=None)

    else:
        series_index    = MetadataField('series_index', is_dc=False,
                                        formatter=float, none_is=1)
    title_sort      = TitleSortField('title_sort', is_dc=False)
    rating          = MetadataField('rating', is_dc=False, formatter=float)
    publication_type = MetadataField('publication_type', is_dc=False)
    timestamp       = MetadataField('timestamp', is_dc=False,
                                    formatter=parse_date, renderer=isoformat)

    user_categories = MetadataField('user_categories', is_dc=False,
                                    formatter=json.loads,
                                    renderer=dump_dict)
    author_link_map = MetadataField('author_link_map', is_dc=False,
                                formatter=json.loads, renderer=dump_dict)

# ... 


























    def fget(self):

        matches = self.authors_path(self.metadata)

        if matches:

            for match in matches:

    # ... 

                if not ans:
                    ans = match.get('file-as', None)

                if ans:
                    return ans



    def fset(self, val):
        matches = self.authors_path(self.metadata)
        if matches:
            for key in matches[0].attrib:
                if key.endswith('file-as'):
                    matches[0].attrib.pop(key)
            matches[0].set('{%s}file-as'%self.NAMESPACES['opf'], unicode(val))

    @dynamic_property

# ... 



    def fget(self):
        ans = []
        for tag in self.tags_path(self.metadata):
            text = self.get_text(tag)
            if text and text.strip():
                ans.extend([x.strip() for x in text.split(',')])
        return ans



    def fset(self, val):
        for tag in list(self.tags_path(self.metadata)):
            tag.getparent().remove(tag)
        for tag in val:
            elem = self.create_metadata_element('subject')
            self.set_text(elem, unicode(tag))

    @dynamic_property

# ... 

    def fget(self):

        ans = None

        for match in self.pubdate_path(self.metadata):

            try:
                val = parse_date(etree.tostring(match, encoding=unicode,
                    method='text', with_tail=False).strip())

            except:
                continue

    # ... 

        return ans

    def fset(self, val):

        least_val = least_elem = None

        for match in self.pubdate_path(self.metadata):

            try:
                cval = parse_date(etree.tostring(match, encoding=unicode,
                    method='text', with_tail=False).strip())

            except:
                match.getparent().remove(match)

            else:
                if not val:
                    match.getparent().remove(match)
                if least_val is None or cval < least_val:
                    least_val, least_elem = cval, match

    # ... 




    def fget(self):
        for match in self.isbn_path(self.metadata):
            return self.get_text(match) or None



    def fset(self, val):
        matches = self.isbn_path(self.metadata)
        if not val:
            for x in matches:
                x.getparent().remove(x)
            return
        if not matches:
            attrib = {'{%s}scheme'%self.NAMESPACES['opf']: 'ISBN'}
            matches = [self.create_metadata_element('identifier',
                                                    attrib=attrib)]
        self.set_text(matches[0], unicode(val))

    @dynamic_property

# ... 




    def fget(self):
        for match in self.application_id_path(self.metadata):
            return self.get_text(match) or None



    def fset(self, val):
        matches = self.application_id_path(self.metadata)
        if not matches:
            attrib = {'{%s}scheme'%self.NAMESPACES['opf']: 'calibre'}
            matches = [self.create_metadata_element('identifier',
                                                    attrib=attrib)]
        self.set_text(matches[0], unicode(val))

    @dynamic_property

# ... 



    def fget(self):
        for match in self.uuid_id_path(self.metadata):
            return self.get_text(match) or None



    def fset(self, val):
        matches = self.uuid_id_path(self.metadata)
        if not matches:
            attrib = {'{%s}scheme'%self.NAMESPACES['opf']: 'uuid'}
            matches = [self.create_metadata_element('identifier',
                                                    attrib=attrib)]
        self.set_text(matches[0], unicode(val))

    @dynamic_property

# ... 




    def fget(self):

        ans = []

        for match in self.languages_path(self.metadata):

            t = self.get_text(match)

    # ... 

        return ans



    def fset(self, val):
        matches = self.languages_path(self.metadata)
        for x in matches:
            x.getparent().remove(x)

        for lang in val:
            l = self.create_metadata_element('language')
            self.set_text(l, unicode(lang))

    @dynamic_property

# ... 




    def fget(self):
        for match in self.bkp_path(self.metadata):
            return self.get_text(match) or None

    @dynamic_property
