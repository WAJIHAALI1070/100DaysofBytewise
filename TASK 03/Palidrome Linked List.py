class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def is_palindrome(head):
    def reverse_linked_list(node):
        prev = None
        curr = node
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev

    if not head or not head.next:
        return True

    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    second_half = reverse_linked_list(slow)

    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True


def print_linked_list(head):
    curr = head
    while curr:
        print(curr.val, end=" -> ")
        curr = curr.next
    print("None")


def create_linked_list():
    values = input("Enter elements of the linked list (space-separated): ").strip().split()
    if not values:
        return None

    head = ListNode(int(values[0]))
    curr = head
    for val in values[1:]:
        curr.next = ListNode(int(val))
        curr = curr.next

    return head


head = create_linked_list()

if is_palindrome(head):
    print("The linked list is a palindrome.")
else:
    print("The linked list is not a palindrome.")
