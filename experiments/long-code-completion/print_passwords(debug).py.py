def print_passwords(request):

    response = HttpResponse(mimetype='application/pdf')
    filename = u'filename=%s.pdf;' % _("passwords")
    response['Content-Disposition'] = filename.encode('utf-8')
    doc = SimpleDocTemplate(response, pagesize=A4, topMargin=-6, bottomMargin=-6, leftMargin=0, rightMargin=0, showBoundary=False)
    story = [Spacer(0,0*cm)]

    data= []
    system_url = config_get("system_url")
    system_welcometext = config_get("system_welcometext")

    for user in User.objects.all().order_by('last_name'):

    # ... 

            user.get_profile()
            cell = []
            cell.append(Spacer(0,0.8*cm))
            cell.append(Paragraph(_("Your Account for OpenSlides"), stylesheet['Ballot_title']))
            cell.append(Paragraph(_("for %s") % (user.profile), stylesheet['Ballot_subtitle']))
            cell.append(Spacer(0,0.5*cm))

            cell.append(Paragraph(_("User: %s") % (user.username), stylesheet['Ballot_option']))
            cell.append(Paragraph(_("Password: %s") % (user.profile.firstpassword), stylesheet['Ballot_option']))
            cell.append(Spacer(0,0.5*cm))
            cell.append(Paragraph(_("URL: %s") % (system_url), stylesheet['Ballot_option']))
            cell.append(Spacer(0,0.5*cm))
            cell2 = []

            cell2.append(Spacer(0,0.8*cm))

            if system_welcometext is not None:
                cell2.append(Paragraph(system_welcometext.replace('\r\n','<br/>'), stylesheet['Ballot_subtitle']))

            data.append([cell,cell2])

    # ... 

    story.append(t)
    doc.build(story)

    return response

# ... 



def print_application_poll(request, poll_id=None):
    poll = Poll.objects.get(id=poll_id)
    response = HttpResponse(mimetype='application/pdf')
    filename = u'filename=%s%s_%s.pdf;' % (_("Application"), str(poll.application.number), _("Poll"))
    response['Content-Disposition'] = filename.encode('utf-8')
    doc = SimpleDocTemplate(response, pagesize=A4, topMargin=-6, bottomMargin=-6, leftMargin=0, rightMargin=0, showBoundary=False)
    story = [Spacer(0,0*cm)]

    imgpath = os.path.join(SITE_ROOT, 'static/images/circle.png')
    circle = "<img src='%s' width='15' height='15'/>&nbsp;&nbsp;" % imgpath
    cell = []
    cell.append(Spacer(0,0.8*cm))
    cell.append(Paragraph(_("Application No.")+" "+str(poll.application.number), stylesheet['Ballot_title']))
    cell.append(Paragraph(poll.application.title, stylesheet['Ballot_subtitle']))
    cell.append(Paragraph(str(poll.ballot)+". "+_("Vote"), stylesheet['Ballot_description']))
    cell.append(Spacer(0,0.5*cm))
    cell.append(Paragraph(circle+_("Yes"), stylesheet['Ballot_option']))
    cell.append(Paragraph(circle+_("No"), stylesheet['Ballot_option']))
    cell.append(Paragraph(circle+_("Abstention"), stylesheet['Ballot_option']))

    data= []
    number = 1
    # get ballot papers config values
    ballot_papers_selection = config_get("application_pdf_ballot_papers_selection")
    ballot_papers_number = config_get("application_pdf_ballot_papers_number")
    # set number of ballot papers
    if ballot_papers_selection == "1":
        number = User.objects.filter(profile__type__iexact="delegate").count()
    if ballot_papers_selection == "2":
        number = int(User.objects.count() - 1)
    if ballot_papers_selection == "0":
        number = int(ballot_papers_number)
    # print ballot papers
    for user in xrange(number/2):
        data.append([cell,cell])
    rest = number % 2
    if rest:
        data.append([cell,''])
    t=Table(data, 10.5*cm, 7.42*cm)
    t.setStyle(TableStyle([ ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                            ('VALIGN', (0,0), (-1,-1), 'TOP'),
                          ]))
    story.append(t)
    doc.build(story)
    return response

# ... 

def get_assignment(assignment, story):
    # title

    story.append(Paragraph(_("Election")+": %s" % assignment.name, stylesheet['Heading1']))
    story.append(Spacer(0,0.5*cm))
    cell1a = []
    cell1a.append(Paragraph("<font name='Ubuntu-Bold'>%s:</font>" % _("Number of available posts"), stylesheet['Bold']))
    cell1b = []

    cell1b.append(Paragraph(str(assignment.posts), stylesheet['Paragraph']))
    cell2a = []
    cell2a.append(Paragraph("<font name='Ubuntu-Bold'>%s:</font><seqreset id='counter'>" % _("Candidates"), stylesheet['Heading4']))
    cell2b = []

    for c in assignment.profile.all():
        cell2b.append(Paragraph("<seq id='counter'/>.&nbsp; %s" % unicode(c), stylesheet['Signaturefield']))

    if assignment.status == "sea":
        for x in range(0,2*assignment.posts):
            cell2b.append(Paragraph("<seq id='counter'/>.&nbsp; __________________________________________",stylesheet['Signaturefield']))

    cell2b.append(Spacer(0,0.2*cm))
    table_votes = []
    results = get_assignment_votes(assignment)
    cell3a = []
    cell3a.append(Paragraph("%s:" % (_("Vote results")), stylesheet['Heading4']))

    # ... 

        if len(results[0]) >= 1:
            cell3a.append(Paragraph("%s %s" % (len(results[0][1]), _("ballots")), stylesheet['Normal']))

        if len(results[0][1]) > 0:

            data_votes = []
            headrow = []
            headrow.append(_("Candidates"))

    # ... 

            data_votes.append(headrow)

            for candidate in results:

                row = []

                if candidate[0][1]:
                    elected = "* "

                else:
                    elected = ""

                c = str(candidate[0][0]).split("(",1)

                if len(c) > 1:
                    row.append(elected+c[0]+"\n"+"("+c[1])

                else:
                    row.append(elected+str(candidate[0][0]))

    # ... 

                    else:
                        row.append(str(votes))

    # ... 

            polls = []

            for poll in assignment.poll_set.filter(assignment=assignment):
                polls.append(poll)

            row = []
            row.append(_("Invalid votes"))

    # ... 

            data_votes.append(row)

            row = []
            row.append(_("Votes cast"))

            for p in polls:
                if p.published:
                    row.append(p.votescastf)

            data_votes.append(row)
            table_votes=Table(data_votes)

    # ... 

    if table_votes:
        data.append([cell3a,table_votes])
        data.append(['','* = '+_('elected')])

    else:
        data.append([cell2a,cell2b])
    data.append([Spacer(0,0.2*cm),''])
    t=Table(data)
    t._argW[0]=4.5*cm

    t._argW[1]=11*cm
    t.setStyle(TableStyle([ ('BOX', (0,0), (-1,-1), 1, colors.black),
                            ('VALIGN', (0,0), (-1,-1), 'TOP'),
                          ]))

    story.append(t)
    story.append(Spacer(0,1*cm))
    # text

    story.append(Paragraph("%s" % assignment.description.replace('\r\n','<br/>'), stylesheet['Paragraph']))

    return story