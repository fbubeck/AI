?&$	s??????$?M?????2??(??!5?Ry;???$	?1??W{@?#?ñ??f??x?.@!??X?
?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?v?$????<e5]??A0F$
-???Yv???;???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails01DN_????F;?I??A|??˙???Y.?&??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails05?Ry;?????? ?S??Au?b?T4??Y??E
e??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0N*k????ȓ?k&??A???x?&??Y?%W??M??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0s??+\???????Z??A?+??E|??Y?? %̔?rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?x??n???5c?tv2??A:;%???Y??????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?#?@????B?Գ ??A?@j'??Y/?o??e??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?_ѭ???Pr?Md???A{??9y???Y??v?????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
˟o???ۿ?Ҥ??Ae?I)????Y߇??(_??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0$?@????ʅʿ???A???0?:??Y??̯? ??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?{,}???T?J?ó??A?uʣ??Yr??9???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?u?T??5??????A<1??PN??Yip[[x??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?2??(??f??
???APoF?W???Y????????rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???LL???g?????Aup?x???Y???^??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????B,??,E??@J??A?@?]????Y?H??rړ?rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??l?%??}?͍?	??A???9????Y?|A??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails02>?^?????Ry=???A?C??{??Y'.?+=??rtrain 97*	c;?O?x?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat=??tZ???!G?@+QB@)g'??????1?\?h?A@:Preprocessing2E
Iterator::Root?30??&??!??QAJG=@)?h????1??s,?-@:Preprocessing2T
Iterator::Root::ParallelMapV2|H??ߠ??!Y?h?,@)|H??ߠ??1Y?h?,@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?a?????!܋?o-?Q@)M?T?#???1??I?)@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?W?\T??!????"@)?W?\T??1????"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?!?uq??!]%??0@):?S?????1Y)ű??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?tw????!???T!5@)UL??pv??1?TV?D&@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*YM?]??!?X't??@)YM?]??1?X't??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??2??j@Ij???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?
??2?????<????5??????!??? ?S??	!       "	!       *	!       2$	wL???>?s?????PoF?W???!???0?:??:	!       B	!       J$	2??.??????>???n?ip[[x??!?|A??R	!       Z$	2??.??????>???n?ip[[x??!?|A??b	!       JCPU_ONLYY??2??j@b qj???X@Y      Y@q?֪??[T@"?	
both?Your program is POTENTIALLY input-bound because 48.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?81.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 