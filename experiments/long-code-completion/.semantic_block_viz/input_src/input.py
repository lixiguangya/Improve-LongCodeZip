def update(self, full=False):

    # Marker
    tStart = time.time()

    # The current time is further adjusted by the 'timeOffset' attribute
    # to help with tracking.
    tNow = datetime.utcnow() + self.timeOffset

    # Update the observer with the current time
    self.observer.date = tNow.strftime("%Y/%m/%d %H:%M:%S.%f")

    # Loop through the satellites and update them
    for tier,sat in zip(self.tiers, self.satellites):
        if not full:
            if tier != self.currentTier and sat.catalog_number != self.tracking:
                continue

        sat.compute(self.observer)

        if sat.alt > 0:
            ## If the satellite is up, check and see if it is 
            ## the one that we should be tracking.
            if sat.catalog_number == self.tracking:
                ### Is there a telescope to use?
                if self.telescope is not None:
                    #### Apply a perpendicular correction to the
                    #### track to help with tracking
                    ra, dec = getPointFromBearing(sat.ra, sat.dec, sat.bearing+math.pi/2, self.crossTrackOffset)

                    #### Radians -> hours/degrees
                    ra = ra*_rad2hr
                    dec = dec*_rad2deg

                    #### Command the telescope
                    self.telescope.moveToPosition(ra, dec, blocking=False)

        else:
            ## If it is no longer visible, check and see if it is
            ## the satellite that we were tracking so that we can
            ## stop.
            if sat.catalog_number == self.tracking:
                self.stopTracking()

    # Final time to figure out how much time we spent calculating 
    # positions.
    tStop = time.time()

    # Update the tier being computed
    self.currentTier = (self.currentTier + 1) % self.nTiers

    # Done
    return tStop-tStart
