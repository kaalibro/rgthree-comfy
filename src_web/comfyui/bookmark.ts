import { RgthreeBaseVirtualNodeConstructor } from "typings/rgthree.js";
// @ts-ignore
import { app } from "../../scripts/app.js";
import { RgthreeBaseVirtualNode } from "./base_node.js";
import { rgthree } from "./rgthree.js";
import { NodeTypesString } from "./constants.js";
import type {
  LGraph as TLGraph,
  LiteGraph as TLiteGraph,
  LGraphCanvas as TLGraphCanvas,
  INumberWidget,
  Vector2,
} from "typings/litegraph.js";
import { getClosestOrSelf, queryOne } from "rgthree/common/utils_dom.js";

declare const LiteGraph: typeof TLiteGraph;

/**
 * A bookmark node. Can be placed anywhere in the workflow, and given a shortcut key that will
 * navigate to that node, with it in the top-left corner.
 */
export class Bookmark extends RgthreeBaseVirtualNode {
  static override type = NodeTypesString.BOOKMARK;
  static override title = NodeTypesString.BOOKMARK;
  override comfyClass = NodeTypesString.BOOKMARK;

  // Really silly, but Litegraph assumes we have at least one input/output... so we need to
  // counteract it's computeSize calculation by offsetting the start.
  static slot_start_y = -20;

  // LiteGraph adds mroe spacing than we want when calculating a nodes' `_collapsed_width`, so we'll
  // override it with a setter and re-set it measured exactly as we want.
  ___collapsed_width: number = 0;

  override isVirtualNode = true;
  override serialize_widgets = true;

  //@ts-ignore - TS Doesn't like us overriding a property with accessors but, too bad.
  override get _collapsed_width() {
    return this.___collapsed_width;
  }

  override set _collapsed_width(width: number) {
    const canvas = app.canvas as TLGraphCanvas;
    const ctx = canvas.canvas.getContext("2d")!;
    const oldFont = ctx.font;
    ctx.font = canvas.title_text_font;
    this.___collapsed_width = 40 + ctx.measureText(this.title).width;
    ctx.font = oldFont;
  }

  readonly keypressBound;

  constructor(title = Bookmark.title) {
    super(title);
    this.addWidget(
      "text",
      "shortcut_key",
      "1",
      (value: string, ...args) => {
        value = value.trim()[0] || "1";
      },
      {
        y: 8,
      },
    );
    this.addWidget<INumberWidget>("number", "zoom", 1, (value: number) => {}, {
      y: 8 + LiteGraph.NODE_WIDGET_HEIGHT + 4,
      max: 2,
      min: 0.5,
      precision: 2,
    });
    this.keypressBound = this.onKeypress.bind(this);
    this.onConstructed();
  }

  // override computeSize(out?: Vector2 | undefined): Vector2 {
  //   super.computeSize(out);
  //   const minHeight = (this.widgets?.length || 0) * (LiteGraph.NODE_WIDGET_HEIGHT + 4) + 16;
  //   this.size[1] = Math.max(minHeight, this.size[1]);
  // }

  get shortcutKey(): string {
    return this.widgets[0]?.value?.toLocaleLowerCase() ?? "";
  }

  override onAdded(graph: TLGraph): void {
    rgthree.addEventListener("keydown", this.keypressBound as EventListener);
  }

  override onRemoved(): void {
    rgthree.removeEventListener("keydown", this.keypressBound as EventListener);
  }

  onKeypress(event: CustomEvent<{ originalEvent: KeyboardEvent }>) {
    const originalEvent = event.detail.originalEvent;
    const target = (originalEvent.target as HTMLElement)!;
    if (getClosestOrSelf(target, 'input,textarea,[contenteditable="true"]')) {
      return;
    }

    // Only the shortcut keys are held down, otionally including "shift".
    if (rgthree.areOnlyKeysDown(this.widgets[0]!.value, true)) {
      this.canvasToBookmark();
      originalEvent.preventDefault();
      originalEvent.stopPropagation();
    }
  }

  /**
   * Called from LiteGraph's `processMouseDown` after it would invoke the input box for the
   * shortcut_key, so we check if it exists and then add our own event listener so we can track the
   * keys down for the user.
   */
  override onMouseDown(event: MouseEvent, pos: Vector2, graphCanvas: TLGraphCanvas): void {
    const input = queryOne<HTMLInputElement>('.graphdialog > input.value');
    if (input && input.value === this.widgets[0]?.value) {
      input.addEventListener('keydown', (e) => {
        // ComfyUI swallows keydown on inputs, so we need to call out to rgthree to use downkeys.
        rgthree.handleKeydown(e);
        e.preventDefault();
        e.stopPropagation();
        input.value = Object.keys(rgthree.downKeys).join(" + ");
      });
    }
  }

  canvasToBookmark() {
    const canvas = app.canvas as TLGraphCanvas;
    // ComfyUI seemed to break us again, but couldn't repro. No reason to not check, I guess.
    // https://github.com/rgthree/rgthree-comfy/issues/71
    if (canvas?.ds?.offset) {
      canvas.ds.offset[0] = -this.pos[0] + 16;
      canvas.ds.offset[1] = -this.pos[1] + 40;
    }
    if (canvas?.ds?.scale != null) {
      canvas.ds.scale = Number(this.widgets[1]!.value || 1);
    }
    canvas.setDirty(true, true);
  }

  static override setUp(clazz: RgthreeBaseVirtualNodeConstructor) {
    LiteGraph.registerNodeType(clazz.type, clazz);
    clazz.category = clazz._category;
  }
}

app.registerExtension({
  name: "rgthree.Bookmark",
  registerCustomNodes() {
    Bookmark.setUp(Bookmark);
  },
});
