class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_linked_list(head):
    prev = None
    current = head

    while current:
        next_node = current.next  # Store the next node
        current.next = prev  # Reverse the pointer

        # Move pointers one position ahead
        prev = current
        current = next_node

    return prev  # New head of the reversed list


def print_linked_list(head):
    current = head
    while current:
        print(current.val, end=" -> ")
        current = current.next
    print("None")


# Function to create a linked list from input values
def create_linked_list(values):
    if not values:
        return None

    head = ListNode(values[0])
    current = head

    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next

    return head
if __name__ == "__main__":
    # Input linked list values
    values = list(map(int, input("Enter space-separated values for linked list: ").split()))

    # Create the linked list from input values
    head = create_linked_list(values)

    # Print original linked list
    print("Original Linked List:")
    print_linked_list(head)

    # Reverse the linked list
    reversed_head = reverse_linked_list(head)

    # Print reversed linked list
    print("\nReversed Linked List:")
    print_linked_list(reversed_head)

