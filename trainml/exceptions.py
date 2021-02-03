class ApiError(Exception):
    def __init__(self, status, data, *args):
        super().__init__(status, data, *args)
        self.status = status
        self.message = data.get("errorMessage")

    def __repr__(self):
        return "ApiError({self.status}, {self.message!r})".format(self=self)

    def __str__(self):
        return "ApiError({self.status}, {self.message!r})".format(self=self)


class JobError(Exception):
    def __init__(self, status, data, *args):
        super().__init__(status, data, *args)
        self.status = status
        self.message = data

    def __repr__(self):
        return "JobError({self.status}, {self.message!r})".format(self=self)

    def __str__(self):
        return "JobError({self.status}, {self.message!r})".format(self=self)


class DatasetError(Exception):
    def __init__(self, status, data, *args):
        super().__init__(status, data, *args)
        self.status = status
        self.message = data

    def __repr__(self):
        return "JobError({self.status}, {self.message!r})".format(self=self)

    def __str__(self):
        return "JobError({self.status}, {self.message!r})".format(self=self)