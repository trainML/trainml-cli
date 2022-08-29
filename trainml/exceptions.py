import logging


class TrainMLException(Exception):
    def __init__(self, message, *args):
        super().__init__(message, *args)
        self.message = message

    def __repr__(self):
        return "TrainMLException( {self.message!r})".format(self=self)

    def __str__(self):
        return "TrainMLException({self.message!r})".format(self=self)


class ApiError(TrainMLException):
    def __init__(self, status, data, *args):
        super().__init__(data, *args)
        self.status = status
        logging.debug(data)
        self.message = data.get("errorMessage") or data.get("message")

    def __repr__(self):
        return "ApiError({self.status}, {self.message!r})".format(self=self)

    def __str__(self):
        return "ApiError({self.status}, {self.message!r})".format(self=self)


class JobError(TrainMLException):
    def __init__(self, status, data, *args):
        super().__init__(data, *args)
        self.status = status
        self.message = data

    def __repr__(self):
        return "JobError({self.status}, {self.message})".format(self=self)

    def __str__(self):
        return "JobError({self.status}, {self.message})".format(self=self)


class DatasetError(TrainMLException):
    def __init__(self, status, data, *args):
        super().__init__(data, *args)
        self.status = status
        self.message = data

    def __repr__(self):
        return "DatasetError({self.status}, {self.message})".format(self=self)

    def __str__(self):
        return "DatasetError({self.status}, {self.message})".format(self=self)


class ModelError(TrainMLException):
    def __init__(self, status, data, *args):
        super().__init__(data, *args)
        self.status = status
        self.message = data

    def __repr__(self):
        return "ModelError({self.status}, {self.message})".format(self=self)

    def __str__(self):
        return "ModelError({self.status}, {self.message})".format(self=self)


class ConnectionError(TrainMLException):
    def __init__(self, message, *args):
        super().__init__(message, *args)
        self.message = message

    def __repr__(self):
        return "ConnectionError({self.message})".format(self=self)

    def __str__(self):
        return "ConnectionError({self.message})".format(self=self)


class SpecificationError(TrainMLException):
    def __init__(self, attribute, message, *args):
        super().__init__(message, *args)
        self.attribute = attribute
        self.message = message

    def __repr__(self):
        return "SpecificationError({self.attribute}, {self.message})".format(
            self=self
        )

    def __str__(self):
        return "SpecificationError({self.attribute}, {self.message})".format(
            self=self
        )
