

def net_install(self,after_download):
    # initialise the profile, from the server if any
    if self.profile:
        profile_data = self.get_data("profile",self.profile)
    elif self.system:
        profile_data = self.get_data("system",self.system)
    elif self.image:
        profile_data = self.get_data("image",self.image)
    else:
        # shouldn't end up here, right?
        profile_data = {}

    if profile_data.get("kickstart","") != "":

        # fix URLs
        if profile_data["kickstart"][0] == "/" or profile_data["template_remote_kickstarts"]:
           if not self.system:
               profile_data["kickstart"] = "http://%s/cblr/svc/op/ks/profile/%s" % (profile_data['http_server'], profile_data['name'])
           else:
               profile_data["kickstart"] = "http://%s/cblr/svc/op/ks/system/%s" % (profile_data['http_server'], profile_data['name'])

        # find_kickstart source tree in the kickstart file
        self.get_install_tree_from_kickstart(profile_data)

        # if we found an install_tree, and we don't have a kernel or initrd
        # use the ones in the install_tree
        if self.safe_load(profile_data,"install_tree"):
            if not self.safe_load(profile_data,"kernel"):
                profile_data["kernel"] = profile_data["install_tree"] + "/images/pxeboot/vmlinuz"

            if not self.safe_load(profile_data,"initrd"):
                profile_data["initrd"] = profile_data["install_tree"] + "/images/pxeboot/initrd.img"


    # find the correct file download location 
    if not self.is_virt:
        if os.path.exists("/boot/efi/EFI/redhat/elilo.conf"):
            # elilo itanium support, may actually still work
            download = "/boot/efi/EFI/redhat"
        else:
            # whew, we have a sane bootloader
            download = "/boot"

    else:
        # ensure we have a good virt type choice and know where
        # to download the kernel/initrd
        if self.virt_type is None:
            self.virt_type = self.safe_load(profile_data,'virt_type',default=None)
        if self.virt_type is None or self.virt_type == "":
            self.virt_type = "auto"

        # if virt type is auto, reset it to a value we can actually use
        if self.virt_type == "auto":

            if profile_data.get("xml_file","") != "":
                raise InfoException("xmlfile based installations are not supported")

            elif profile_data.has_key("file"):
                print "- ISO or Image based installation, always uses --virt-type=qemu"
                self.virt_type = "qemu"

            else:
                # FIXME: auto never selects vmware, maybe it should if we find it?

                if not ANCIENT_PYTHON:
                    cmd = sub_process.Popen("/bin/uname -r", stdout=sub_process.PIPE, shell=True)
                    uname_str = cmd.communicate()[0]
                    if uname_str.find("xen") != -1:
                        self.virt_type = "xenpv"
                    elif os.path.exists("/usr/bin/qemu-img"):
                        self.virt_type = "qemu"
                    else:
                        # assume Xen, we'll check to see if virt-type is really usable later.
                        raise InfoException, "Not running a Xen kernel and qemu is not installed"

            print "- no virt-type specified, auto-selecting %s" % self.virt_type

        # now that we've figured out our virt-type, let's see if it is really usable
        # rather than showing obscure error messages from Xen to the user :)

        if self.virt_type in [ "xenpv", "xenfv" ]:
            cmd = sub_process.Popen("uname -r", stdout=sub_process.PIPE, shell=True)
            uname_str = cmd.communicate()[0]
            # correct kernel on dom0?
            if uname_str.find("xen") == -1:
               raise InfoException("kernel-xen needs to be in use")
            # xend installed?
            if not os.path.exists("/usr/sbin/xend"):
               raise InfoException("xen package needs to be installed")
            # xend running?
            rc = sub_process.call("/usr/sbin/xend status", stderr=None, stdout=None, shell=True)
            if rc != 0:
               raise InfoException("xend needs to be started")

        # for qemu
        if self.virt_type == "qemu":
            # qemu package installed?
            if not os.path.exists("/usr/bin/qemu-img"):
                raise InfoException("qemu package needs to be installed")
            # is libvirt new enough?
            cmd = sub_process.Popen("rpm -q python-virtinst", stdout=sub_process.PIPE, shell=True)
            version_str = cmd.communicate()[0]
            if version_str.find("virtinst-0.1") != -1 or version_str.find("virtinst-0.0") != -1:
                raise InfoException("need python-virtinst >= 0.2 to do installs for qemu/kvm")

        # for vmware
        if self.virt_type == "vmware" or self.virt_type == "vmwarew":
            # FIXME: if any vmware specific checks are required (for deps) do them here.
            pass

        if self.virt_type == "virt-image":
            if not os.path.exists("/usr/bin/virt-image"):
                raise InfoException("virt-image not present, downlevel virt-install package?")

        # for both virt types
        if os.path.exists("/etc/rc.d/init.d/libvirtd"):
            rc = sub_process.call("/sbin/service libvirtd status", stdout=None, shell=True)
            if rc != 0:
                # libvirt running?
                raise InfoException("libvirtd needs to be running")


        if self.virt_type in [ "xenpv" ]:
            # we need to fetch the kernel/initrd to do this
            download = "/var/lib/xen" 
        elif self.virt_type in [ "xenfv", "vmware", "vmwarew" ] :
            # we are downloading sufficient metadata to initiate PXE, no D/L needed
            download = None 
        else: # qemu
            # fullvirt, can use set_location in virtinst library, no D/L needed yet
            download = None 

    # download required files
    if not self.is_display and download is not None:
       self.get_distro_files(profile_data, download)

    # perform specified action
    after_download(self, profile_data)
