$	?*?Z3?????c? ???aL?{)??!??p?????$	? ?$@?р?Y???S?0Y?
	@!ibvʡ@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??p???????c?????AE.8??_??Y?H/j????rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?۽?'G?????y7??AZ)r?#??Y??L?*???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Eg?E(???N??ĭ??A?[????Y?2SZK??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0B?Ѫ?t???Y?N???AC?ʠ????Y:#/kb??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0B?!????U?g$B#??AR_?vj.??Y???8Q??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??y7??D?X?oC??A2??Y???Y?2?&c??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0f???-=??ƈD?e??A??eO??Y?={.S???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?aL?{)?????ϝ`??Ah%??????Y??c?~??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
???m????#??????Av4?????Y?8??m4??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0m??~????M?D?u???A?z0)>>??Y?T???B??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?GR??????0`?U,??A|??l;m??Y?26t????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?$?????;??.??A?O ????Y?????Đ?rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ӝ'????Pj???A???<,???YGUD???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?%z????7???-??A?`????Y}??A?<??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?v??-u????M+???A??Đ?L??Y?S㥛Đ?rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0jh????/?x????A??%jj??YXs?`???rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????????????A?????Y?@?]????rtrain 97*	??????@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatE?J?E??!?(??A@)???-??1??D@??@:Preprocessing2E
Iterator::Rootm???????!=)?VI@@)???8+???140?q?0@:Preprocessing2T
Iterator::Root::ParallelMapV2T?d???!?<?B?.@)T?d???1?<?B?.@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipӠh?"??!ak?T??P@)??????1|E??i?'@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?4-?2??!'0A?y<#@)?4-?2??1'0A?y<#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?j?=&R??!R??M?c0@)???M?q??1?$Hq?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*gaO;?5??!?????@)gaO;?5??1?????@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(??G???!F}?5?w4@)??JY?8??1?o_??P@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9bbW?@Im?D?kX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?9?˪???\YE?S?????ϝ`??!??c?????	!       "	!       *	!       2$	?!???Y??b???h%??????!Z)r?#??:	!       B	!       J$	?H??d????Vp?c?Xs?`???!??c?~??R	!       Z$	?H??d????Vp?c?Xs?`???!??c?~??b	!       JCPU_ONLYYbbW?@b qm?D?kX@